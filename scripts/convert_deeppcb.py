#!/usr/bin/env python3
"""Convert official DeepPCB splits into Florence-2 ``<OD>`` JSONL records.

Only ``trainval.txt`` and ``test.txt`` are used.  DeepPCB split entries use a
logical ``123.jpg`` name; the defective input image on disk is ``123_test.jpg``.
Coordinates are encoded on Florence-2's inclusive 0--999 location-token grid.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_NAMES = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}
NAME_TO_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}
LOCATION_RE = re.compile(r"(open|short|mousebite|spur|copper|pin-hole)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
COLORS = ("red", "lime", "cyan", "yellow", "magenta", "orange")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int


def resolve_test_image(split_image_path: Path) -> Path:
    return split_image_path if split_image_path.stem.endswith("_test") else split_image_path.with_name(f"{split_image_path.stem}_test{split_image_path.suffix}")


def read_split(path: Path) -> list[tuple[Path, Path]]:
    records = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected image and annotation path")
        records.append((resolve_test_image(Path(fields[0])), Path(fields[1])))
    return records


def read_boxes(path: Path) -> list[Box]:
    boxes = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = [item for item in re.split(r"[\s,]+", raw_line.strip()) if item]
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 values")
        values = [int(item) for item in fields]
        if values[4] not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number}: invalid class {values[4]}")
        boxes.append(Box(*values))
    return boxes


def to_location(value: int, dimension: int) -> int:
    return max(0, min(999, round(value / dimension * 999)))


def from_location(token: int, dimension: int) -> int:
    return max(0, min(dimension, round(token / 999 * dimension)))


def encode_boxes(boxes: list[Box], width: int, height: int) -> str:
    return "".join(
        f"{CLASS_NAMES[box.class_id]}<loc_{to_location(box.x1, width)}><loc_{to_location(box.y1, height)}>"
        f"<loc_{to_location(box.x2, width)}><loc_{to_location(box.y2, height)}>"
        for box in boxes
    )


def decode_suffix(suffix: str, width: int, height: int) -> list[Box]:
    matches = list(LOCATION_RE.finditer(suffix))
    if "".join(match.group(0) for match in matches) != suffix:
        raise ValueError(f"unparseable suffix: {suffix}")
    return [
        Box(from_location(int(match.group(2)), width), from_location(int(match.group(3)), height), from_location(int(match.group(4)), width), from_location(int(match.group(5)), height), NAME_TO_ID[match.group(1)])
        for match in matches
    ]


def draw_visualization(image_path: Path, original: list[Box], restored: list[Box], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    drawing = ImageDraw.Draw(image)
    for original_box, restored_box in zip(original, restored):
        color = COLORS[original_box.class_id - 1]
        drawing.rectangle((original_box.x1, original_box.y1, original_box.x2, original_box.y2), outline=color, width=3)
        # Cyan inner line is the decoded token-coordinate box. It should nearly overlap.
        drawing.rectangle((restored_box.x1, restored_box.y1, restored_box.x2, restored_box.y2), outline="white", width=1)
        drawing.text((original_box.x1, max(0, original_box.y1 - 12)), CLASS_NAMES[original_box.class_id], fill=color)
    image.save(output_path, quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visualizations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root, output_dir = args.dataset_root.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": read_split(dataset_root / "trainval.txt"), "test": read_split(dataset_root / "test.txt")}
    split_images = {name: {image for image, _ in records} for name, records in splits.items()}
    overlap = split_images["train"] & split_images["test"]
    if overlap:
        raise RuntimeError(f"train/test leakage: {sorted(overlap)[:5]}")

    all_samples = []
    errors: list[float] = []
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    for split_name, records in splits.items():
        destination = output_dir / f"{split_name}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for image_relative, annotation_relative in records:
                image_path, annotation_path = dataset_root / image_relative, dataset_root / annotation_relative
                if not image_path.is_file() or not annotation_path.is_file():
                    raise FileNotFoundError(f"missing pair: {image_path}, {annotation_path}")
                with Image.open(image_path) as image:
                    width, height = image.size
                original = read_boxes(annotation_path)
                suffix = encode_boxes(original, width, height)
                restored = decode_suffix(suffix, width, height)
                if len(original) != len(restored) or [box.class_id for box in original] != [box.class_id for box in restored]:
                    raise AssertionError(f"class/order round-trip failed: {annotation_path}")
                for source, decoded in zip(original, restored):
                    errors.extend((abs(source.x1 - decoded.x1), abs(source.y1 - decoded.y1), abs(source.x2 - decoded.x2), abs(source.y2 - decoded.y2)))
                    class_counts[CLASS_NAMES[source.class_id]] += 1
                handle.write(json.dumps({"image": str(image_path), "prefix": "<OD>", "suffix": suffix}, ensure_ascii=False) + "\n")
                all_samples.append((split_name, image_path, original, restored))

    (output_dir / "class_names.json").write_text(json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(exist_ok=True)
    selected = random.Random(args.seed).sample(all_samples, k=min(args.visualizations, len(all_samples)))
    visualization_paths = []
    for index, (split_name, image_path, original, restored) in enumerate(selected, 1):
        output_path = visualization_dir / f"{index:02d}_{split_name}_{image_path.name}"
        draw_visualization(image_path, original, restored, output_path)
        visualization_paths.append(str(output_path))

    report = {
        "dataset_root": str(dataset_root),
        "records": {split: len(records) for split, records in splits.items()},
        "train_test_overlap": len(overlap),
        "input_images_are_test_images_only": all(path.name.endswith("_test.jpg") for samples in splits.values() for path, _ in samples),
        "class_counts": class_counts,
        "coordinate_mapping": {"range": [0, 999], "encode": "round(x / dimension * 999), clamped", "decode": "round(token / 999 * dimension), clamped"},
        "round_trip": {"boxes_checked": sum(len(sample[2]) for sample in all_samples), "coordinate_values_checked": len(errors), "max_absolute_error_pixels": max(errors, default=0), "mean_absolute_error_pixels": sum(errors) / len(errors) if errors else 0, "within_one_pixel": max(errors, default=0) <= 1},
        "visualizations": visualization_paths,
        "excluded_unreferenced_template": "group90100/90100/90100034_temp.jpg",
    }
    (output_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
