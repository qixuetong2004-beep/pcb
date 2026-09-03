#!/usr/bin/env python3
"""Run the existing Florence-2 app detector on a preprocessed PNG tile/image."""
import argparse, json
from pathlib import Path
from PIL import Image

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('image', type=Path); ap.add_argument('--output-dir', type=Path, default=Path('outputs/preprocessed_detection')); ap.add_argument('--max-new-tokens', type=int, default=512)
    a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    # app.detect is the same code path used by Gradio and preserves raw VLM text.
    from app import detect
    canvas, prompt, raw, rows, report, _ = detect(Image.open(a.image).convert('RGB'), a.max_new_tokens)
    stem = a.image.stem; canvas.save(a.output_dir/f'{stem}_detected.png')
    payload = {'image': str(a.image), 'task_prompt': prompt, 'raw_vlm_sequence': raw, 'table': rows, 'chinese_report': report}
    (a.output_dir/f'{stem}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
