#!/usr/bin/env python3
"""Small, bounded LoRA overfit check for the converted Florence-2 data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/overfit_smoke"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    records = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines()][:args.samples]
    if not 8 <= len(records) <= 16:
        raise ValueError("--samples must select 8--16 records")
    device, dtype = "cuda:0", torch.float16
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, torch_dtype=dtype).to(device)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(base, config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    losses = []
    model.train()
    for step in range(args.steps):
        record = records[step % len(records)]
        image = Image.open(record["image"]).convert("RGB")
        inputs = processor(text=record["prefix"], images=image, return_tensors="pt").to(device, dtype)
        labels = processor.tokenizer(text=record["suffix"], return_tensors="pt", return_token_type_ids=False).input_ids.to(device)
        loss = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], labels=labels).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0: print(f"step={step + 1} loss={losses[-1]:.5f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "adapter")
    processor.save_pretrained(args.output_dir / "adapter")
    report = {"samples": len(records), "steps": args.steps, "first_loss": losses[0], "last_loss": losses[-1], "minimum_loss": min(losses), "loss_decreased": losses[-1] < losses[0], "seed": args.seed}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
