#!/usr/bin/env python3
"""Interactive Florence-2 PCB defect demo with visible VLM text output."""
import json
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/Florence-2-base-ft"
ADAPTER = Path("outputs/full_training/best_adapter")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
CHINESE = {"open": "断路", "short": "短路", "mousebite": "鼠咬", "spur": "毛刺", "copper": "多余铜", "pin-hole": "针孔"}
COLORS = {"open": "#ef4444", "short": "#22c55e", "mousebite": "#06b6d4", "spur": "#eab308", "copper": "#d946ef", "pin-hole": "#f97316"}
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
processor = model = None

def load_model():
    global processor, model
    if model is None:
        processor = AutoProcessor.from_pretrained(ADAPTER, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=DTYPE)
        model = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE).eval()

def region(box, size):
    x, y = (box[0] + box[2]) / 2 / size[0], (box[1] + box[3]) / 2 / size[1]
    horizontal = "左侧" if x < .33 else "右侧" if x > .67 else "中心"
    vertical = "上方" if y < .33 else "下方" if y > .67 else "中部"
    return horizontal + vertical

def report(predictions, size):
    if not predictions: return "未检测到可解析的 PCB 缺陷。"
    counts = Counter(p["label"] for p in predictions)
    summary = "，".join(f"{CHINESE[k]} {v} 处" for k, v in counts.items())
    details = "；".join(f"{CHINESE[p['label']]}位于{region(p['bbox'], size)}" for p in predictions)
    return f"检测到 {len(predictions)} 处 PCB 缺陷：{summary}。{details}。"

def detect(image, max_tokens):
    if image is None: raise gr.Error("请先上传一张 PCB 图片")
    load_model(); image = image.convert("RGB")
    prompt = "<OD>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=int(max_tokens), num_beams=3, do_sample=False, return_dict_in_generate=True, output_scores=True)
    raw = processor.batch_decode(output.sequences, skip_special_tokens=False)[0]
    try:
        parsed = processor.post_process_generation(raw, task=prompt, image_size=image.size)[prompt]
        confidence = float(output.sequences_scores[0]) if output.sequences_scores is not None else None
        predictions = [{"label": label, "bbox": [round(v) for v in box], "confidence": confidence} for label, box in zip(parsed.get("labels", []), parsed.get("bboxes", [])) if label in CHINESE]
    except Exception:
        predictions = []
    canvas = image.copy(); drawing = ImageDraw.Draw(canvas)
    for p in predictions:
        drawing.rectangle(p["bbox"], outline=COLORS[p["label"]], width=3)
        drawing.text((p["bbox"][0], max(0, p["bbox"][1] - 22)), CHINESE[p["label"]], font=FONT, fill=COLORS[p["label"]])
    rows = [[CHINESE[p["label"]], *p["bbox"], region(p["bbox"], image.size)] for p in predictions]
    payload = {"task_prompt": prompt, "raw_vlm_sequence": raw, "detections": predictions, "chinese_report": report(predictions, image.size), "report_note": "中文报告由检测结果按规则生成，不代表模型从 DeepPCB 学到维修知识。"}
    path = Path(tempfile.mkdtemp()) / "pcb_detection.json"; path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Gradio 5 validates File output as a string, not pathlib.Path.
    return canvas, prompt, raw, rows, payload["chinese_report"], str(path)

with gr.Blocks(title="Florence-2 PCB 缺陷检测") as demo:
    gr.Markdown("# Florence-2 PCB 缺陷检测与智能描述\n模型生成位置 token 文本，再解析为边界框；下方保留原始 VLM 输出以便演示。")
    with gr.Row():
        image = gr.Image(type="pil", label="上传 PCB 图片")
        result = gr.Image(label="检测可视化（中文标签）")
    tokens = gr.Slider(64, 512, value=512, step=32, label="最大生成 token 数")
    button = gr.Button("开始 Florence-2 检测", variant="primary")
    prompt = gr.Textbox(label="任务提示词")
    raw = gr.Textbox(label="Florence-2 原始生成序列（VLM 输出）", lines=4)
    table = gr.Dataframe(headers=["类别", "x1", "y1", "x2", "y2", "位置"], label="解析后的结构化检测结果")
    chinese_report = gr.Textbox(label="中文检测报告（规则生成）", lines=4)
    download = gr.File(label="下载 JSON 结果")
    button.click(detect, [image, tokens], [result, prompt, raw, table, chinese_report, download])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
