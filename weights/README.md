# 四组对比实验权重

| 排名 | 模型 | F1 | mAP@0.5 | 推理时间 | 权重路径 |
|---:|---|---:|---:|---:|---|
| 1 | OpenCV + Florence-2 | 0.929 | 0.963 | 2.4 ms/张 | `florence_opencv/best_adapter/` |
| 2 | OpenCV + YOLO | 0.855 | 0.911 | 2.4 ms/张 | `yolo_opencv/best.pt` |
| 3 | 纯 Florence-2 | 0.509 | 0.337 | 151.5 ms/张 | `florence_original/best_adapter/` |
| 4 | 纯 YOLO | 0.361 | 0.180 | 156.7 ms/张 | `yolo_original/best.pt` |

## 说明

- YOLO 目录中的 `best.pt` 是可直接加载的最佳权重。
- Florence-2 使用 LoRA adapter，加载时需与基础模型 `microsoft/Florence-2-base-ft` 组合使用。
- adapter 目录中的配置、tokenizer 和 `adapter_model.safetensors` 需保持在同一目录中。

## 原始来源

- `yolo_original/best.pt` 来自 `pcb_florence2/weights/experiments/yolo_original/weights/best.pt`。
- `yolo_opencv/best.pt` 来自 `pcb_florence2/weights/experiments/yolo_opencv/weights/best.pt`。
- `florence_original/best_adapter/` 来自 `pcb_florence2/outputs/full_training/best_adapter/`。
- `florence_opencv/best_adapter/` 来自 `pcb_florence2/outputs/florence_opencv_training/best_adapter/`。
