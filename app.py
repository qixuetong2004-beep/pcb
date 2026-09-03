#!/usr/bin/env python3
"""Interactive Florence-2 PCB defect demo with visible VLM text output."""
import json
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
import torch
import cv2
import numpy as np
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor
from src.report_generator import generate_report

MODEL_ID = "microsoft/Florence-2-base-ft"
ADAPTER = Path("outputs/full_training/best_adapter")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
CHINESE = {"open": "断路", "short": "短路", "mousebite": "鼠咬", "spur": "毛刺", "copper": "多余铜", "pin-hole": "针孔"}
COLORS = {"open": "#ef4444", "short": "#22c55e", "mousebite": "#06b6d4", "spur": "#eab308", "copper": "#d946ef", "pin-hole": "#f97316"}
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 25)
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

def preprocess_for_vlm(image):
    """Keep the complete upload, resize to 640 square, and make black-line/white-background image."""
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    arr = cv2.resize(arr, (640, 640), interpolation=cv2.INTER_AREA)
    gray0 = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(gray0, (3, 3), 0)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoise)
    threshold, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = np.where(contrast < threshold, 0, 255).astype(np.uint8)
    border = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    if float((border == 0).mean()) > 0.5:
        out = 255 - out
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    stages = [
        ("原图（缩放）", cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)),
        ("灰度化", gray0), ("去噪", denoise), ("对比度增强", contrast),
        ("Otsu二值化", thresh), ("形态学处理", morph), ("最终模型输入", morph)
    ]
    return Image.fromarray(morph).convert("RGB"), [Image.fromarray(x).convert("RGB") for _, x in stages], [n for n, _ in stages]

def detect(image, max_tokens):
    if image is None: raise gr.Error("请先上传一张 PCB 图片")
    load_model(); image = image.convert("RGB")
    processed, stages, stage_names = preprocess_for_vlm(image)
    prompt = "<OD>"
    inputs = processor(text=prompt, images=processed, return_tensors="pt").to(DEVICE, DTYPE)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=int(max_tokens), num_beams=3, do_sample=False, return_dict_in_generate=True, output_scores=True)
    raw = processor.batch_decode(output.sequences, skip_special_tokens=False)[0]
    try:
        parsed = processor.post_process_generation(raw, task=prompt, image_size=processed.size)[prompt]
        raw_score = float(output.sequences_scores[0]) if output.sequences_scores is not None else 0.0
        # Florence-2 exposes a sequence score rather than calibrated box scores;
        # sigmoid maps it to a readable confidence proxy for conservative reporting.
        confidence = 1.0 / (1.0 + np.exp(-raw_score))
        predictions = [{"label": label, "bbox": [round(v) for v in box], "confidence": confidence} for label, box in zip(parsed.get("labels", []), parsed.get("bboxes", [])) if label in CHINESE]
    except Exception:
        predictions = []
    canvas = processed.copy(); drawing = ImageDraw.Draw(canvas)
    for p in predictions:
        drawing.rectangle(p["bbox"], outline=COLORS[p["label"]], width=3)
        drawing.text((p["bbox"][0], max(0, p["bbox"][1] - 22)), CHINESE[p["label"]], font=FONT, fill=COLORS[p["label"]])
    chinese_report, report_items, report_meta = generate_report(predictions, processed.size)
    rows = [[CHINESE[p["label"]], *p["bbox"], region(p["bbox"], processed.size), round(p["confidence"], 3)] for p in predictions]
    payload = {"task_prompt": prompt, "preprocessing": "resize 640x640 + grayscale + CLAHE + Otsu + morphology", "raw_vlm_sequence": raw, "detections": predictions, "standardized_report": report_items, "report_meta": report_meta, "chinese_report": chinese_report, "report_note": "置信度为Florence-2生成序列分数映射得到的参考值，不等同于校准后的目标级概率；报告由检测结果按规则生成。"}
    path = Path(tempfile.mkdtemp()) / "pcb_detection.json"; path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Gradio 5 validates File output as a string, not pathlib.Path.
    # Gradio Gallery accepts (image, caption) pairs, so every stage is labeled.
    labeled_stages = list(zip(stages, stage_names))
    return processed, canvas, prompt, raw, rows, payload["chinese_report"], labeled_stages, str(path)

UI_CSS = """
.gradio-container { font-size: 17px !important; }
textarea, input { font-size: 17px !important; }
table { font-size: 16px !important; }
"""
with gr.Blocks(title="Florence-2 PCB 缺陷检测", css=UI_CSS) as demo:
    gr.Markdown("""
<div style="text-align:center;">
  <div style="font-size:32px;font-weight:700;">Florence-2 PCB 缺陷检测与智能描述</div>
  <div style="font-size:20px;margin-top:10px;">作者：祁雪桐　　学号：23200202</div>
  <div style="font-size:16px;margin-top:10px;color:#666;">模型生成位置 token 文本，再解析为边界框；下方保留原始 VLM 输出以便演示。</div>
</div>
""")
    with gr.Row():
        image = gr.Image(type="pil", label="上传 PCB 图片")
        with gr.Column():
            processed_result = gr.Image(label="OpenCV 预处理结果（模型输入）")
            result = gr.Image(label="Florence-2 检测可视化（中文标签）")
    tokens = gr.Slider(64, 512, value=512, step=32, label="最大生成 token 数")
    button = gr.Button("开始 Florence-2 检测", variant="primary")
    prompt = gr.Textbox(label="任务提示词")
    raw = gr.Textbox(label="Florence-2 原始生成序列（VLM 输出）", lines=4)
    table = gr.Dataframe(headers=["类别", "x1", "y1", "x2", "y2", "位置", "置信度"], label="解析后的结构化检测结果")
    chinese_report = gr.Textbox(label="中文检测报告（规则生成）", lines=4)
    gr.Markdown("### OpenCV 预处理各阶段")
    stages_gallery = gr.Gallery(label="原图 → 灰度 → 去噪 → 增强 → 二值化 → 形态学 → 模型输入", columns=4, rows=2, height="auto", object_fit="contain")
    download = gr.File(label="下载 JSON 结果")
    button.click(detect, [image, tokens], [processed_result, result, prompt, raw, table, chinese_report, stages_gallery, download])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
