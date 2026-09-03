#!/usr/bin/env python3
"""Validate the original DeepPCB dataset without modifying it.

The official split files contain ``image_relative_path annotation_relative_path``
per line.  Current repository annotations are whitespace-separated, while the
README documents comma-separated values, so both forms are accepted.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}
COLORS = ("red", "lime", "cyan", "yellow", "magenta", "orange")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int


def parse_annotation(path: Path) -> tuple[list[Box], list[str]]:
    boxes: list[Box] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part for part in re.split(r"[\s,]+", line) if part]
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            values = [int(value) for value in parts]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-integer value")
            continue
        boxes.append(Box(*values))
    return boxes, errors


def read_split(path: Path) -> list[tuple[Path, Path]]:
    records: list[tuple[Path, Path]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected image and annotation path")
        records.append((Path(fields[0]), Path(fields[1])))
    return records


def resolve_test_image(split_image_path: Path) -> Path:
    """Map DeepPCB's logical ``123.jpg`` split entry to ``123_test.jpg``."""
    if split_image_path.stem.endswith("_test"):
        return split_image_path
    return split_image_path.with_name(f"{split_image_path.stem}_test{split_image_path.suffix}")


def resolve_template_image(test_image_path: Path) -> Path:
    return test_image_path.with_name(f"{test_image_path.stem.removesuffix('_test')}_temp{test_image_path.suffix}")


def draw_examples(dataset_root: Path, examples, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, (image_relative, boxes) in enumerate(examples, 1):
        image = Image.open(dataset_root / image_relative).convert("RGB")
        drawing = ImageDraw.Draw(image)
        for box in boxes:
            color = COLORS[box.class_id - 1]
            drawing.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)
            drawing.text((box.x1, max(0, box.y1 - 12)), CLASS_NAMES[box.class_id], fill=color)
        output_path = output_dir / f"annotation_example_{index:02d}_{image_relative.stem}.jpg"
        image.save(output_path, quality=95)
        paths.append(str(output_path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="DeepPCB/PCBData directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    splits = {name: read_split(dataset_root / f"{name}.txt") for name in ("trainval", "test")}
    referenced_images = {resolve_test_image(image) for records in splits.values() for image, _ in records}
    referenced_templates = {resolve_template_image(image) for image in referenced_images}
    referenced_annotations = {annotation for records in splits.values() for _, annotation in records}
    discovered_test_images = set(path.relative_to(dataset_root) for path in dataset_root.rglob("*_test.jpg"))
    discovered_templates = set(path.relative_to(dataset_root) for path in dataset_root.rglob("*_temp.jpg"))
    discovered_annotations = {
        path.relative_to(dataset_root)
        for path in dataset_root.rglob("*.txt")
        if path.name not in {"trainval.txt", "test.txt"}
    }

    class_counts: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    issues = {"missing_images": [], "missing_templates": [], "missing_annotations": [], "parse_errors": [], "invalid_boxes": [], "empty_annotations": []}
    parsed_records = []
    for split_name, records in splits.items():
        for split_image_relative, annotation_relative in records:
            image_relative = resolve_test_image(split_image_relative)
            image_path, annotation_path = dataset_root / image_relative, dataset_root / annotation_relative
            if not image_path.is_file():
                issues["missing_images"].append(str(image_relative))
                continue
            if not (dataset_root / resolve_template_image(image_relative)).is_file():
                issues["missing_templates"].append(str(resolve_template_image(image_relative)))
            if not annotation_path.is_file():
                issues["missing_annotations"].append(str(annotation_relative))
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            image_sizes[f"{width}x{height}"] += 1
            boxes, errors = parse_annotation(annotation_path)
            issues["parse_errors"].extend(errors)
            if not boxes:
                issues["empty_annotations"].append(str(annotation_relative))
            for box in boxes:
                if box.class_id not in CLASS_NAMES or box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height or box.x1 >= box.x2 or box.y1 >= box.y2:
                    issues["invalid_boxes"].append(f"{annotation_relative}: {box}")
                elif not errors:
                    class_counts[CLASS_NAMES[box.class_id]] += 1
            parsed_records.append((split_name, image_relative, boxes))

    train_images = {resolve_test_image(image) for image, _ in splits["trainval"]}
    test_images = {resolve_test_image(image) for image, _ in splits["test"]}
    rng = random.Random(args.seed)
    eligible = [(image, boxes) for _, image, boxes in parsed_records if boxes]
    examples = rng.sample(eligible, k=min(args.examples, len(eligible)))
    image_paths = draw_examples(dataset_root, examples, output_dir / "visualizations")

    report = {
        "dataset_root": str(dataset_root),
        "splits": {name: len(records) for name, records in splits.items()},
        "split_overlap_images": len(train_images & test_images),
        "referenced": {"images": len(referenced_images), "annotations": len(referenced_annotations)},
        "discovered": {"test_jpg_images": len(discovered_test_images), "template_jpg_images": len(discovered_templates), "annotation_txt": len(discovered_annotations)},
        "unreferenced": {
            "test_images": sorted(map(str, discovered_test_images - referenced_images)),
            "template_images": sorted(map(str, discovered_templates - referenced_templates)),
            "annotations": sorted(map(str, discovered_annotations - referenced_annotations)),
        },
        "image_sizes": dict(sorted(image_sizes.items())),
        "class_counts": {CLASS_NAMES[index]: class_counts[CLASS_NAMES[index]] for index in CLASS_NAMES},
        "total_boxes": sum(class_counts.values()),
        "issues": issues,
        "visualizations": image_paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    issue_counts = {name: len(values) for name, values in issues.items()}
    markdown = [
        "# DeepPCB 数据检查报告",
        "",
        f"- 数据根目录：`{dataset_root}`",
        f"- 官方划分：trainval {len(splits['trainval'])}，test {len(splits['test'])}，图像交集 {len(train_images & test_images)}",
        f"- 已引用：{len(referenced_images)} 张测试图、{len(referenced_templates)} 张模板图、{len(referenced_annotations)} 个标注；发现 {len(discovered_test_images)} 个测试 JPG、{len(discovered_templates)} 个模板 JPG、{len(discovered_annotations)} 个标注 TXT",
        f"- 图像尺寸：{', '.join(f'{size} × {count}' for size, count in sorted(image_sizes.items()))}",
        f"- 缺陷框总数：{sum(class_counts.values())}",
        "",
        "## 类别框数量",
        "",
    ]
    markdown.extend(f"- {CLASS_NAMES[index]}（{index}）：{class_counts[CLASS_NAMES[index]]}" for index in CLASS_NAMES)
    markdown.extend(["", "## 完整性检查", ""])
    markdown.extend(f"- {name}：{count}" for name, count in issue_counts.items())
    markdown.extend(["", "## 可视化样本", ""])
    markdown.extend(f"- `{path}`" for path in image_paths)
    (output_dir / "dataset_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "dataset_report.json"), "issue_counts": issue_counts}, ensure_ascii=False))
    return 1 if any(issue_counts.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
