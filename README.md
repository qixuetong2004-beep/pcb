# Florence-2 PCB 缺陷检测与智能描述系统

基于 `microsoft/Florence-2-base-ft` 和 LoRA 微调的 PCB 缺陷检测课程项目。系统将上传的彩色 PCB 图像经过 OpenCV 预处理后，交给 Florence-2 生成缺陷类别和位置 token，再解析为边界框、统计结果和中文检测报告。

作者：祁雪桐　　学号：23200202

## 功能

- 上传本地 PCB 图片，不使用真实相机。
- OpenCV 预处理：缩放、灰度化、去噪、CLAHE 增强、Otsu 二值化和形态学处理。
- Florence-2 `<OD>` 目标检测：生成类别文本和 `<loc_...>` 坐标 token。
- 输出检测框、类别、坐标、参考置信度、数量统计和原始 VLM 序列。
- 自动生成中文标准化报告，并展示每个预处理阶段。
- 下载 JSON 检测结果。

## 环境

- Ubuntu 22.04
- Conda 环境：`pcb`
- Python 3.10
- NVIDIA GPU、CUDA、PyTorch 2.11.0
- `transformers==4.49.0`、`accelerate==1.14.0`、`peft==0.20.0`
- OpenCV、NumPy、Gradio 5

## 启动界面

```bash
conda activate pcb
cd /home/qiyue/图像处理/pcb_florence2
env -u ALL_PROXY -u all_proxy \
  HTTP_PROXY=http://127.0.0.1:7897 \
  HTTPS_PROXY=http://127.0.0.1:7897 \
  PYTHONPATH=/home/qiyue/miniconda3/envs/pcb/lib/python3.10/site-packages \
  /home/qiyue/miniconda3/envs/pcb/bin/python app.py
```

浏览器访问：`http://127.0.0.1:7860`

## 数据处理

DeepPCB 原始数据位于 `data/raw/DeepPCB/`，Florence-2 JSONL 标注位于 `data/processed/`，固定划分位于 `data/splits/`。Mendeley 处理图片位于项目旁的 `/home/qiyue/图像处理/mendeley_pcb_processed/`，其中 `original/` 和 `processed/` 文件名一一对应。

批量预处理命令：

```bash
python scripts/preprocess_mendeley.py \
  --input-dir "data/raw/mendeley_pcb_defect/PCB-Defect An Annotated Dataset for Surface Defect/extracted/PCB_Defect/images" \
  --limit 0 --resize-size 640 --flat \
  --output-dir /home/qiyue/图像处理/mendeley_pcb_processed
```

## 训练、评估与单图推理

```bash
python src/train.py --config configs/full_training.json --epochs 10
python src/evaluate.py --index data/splits/test.jsonl --adapter weights/best_adapter --output-dir outputs/full_training/test_predictions --max-new-tokens 512
python scripts/detect_preprocessed.py /home/qiyue/图像处理/mendeley_pcb_processed/processed/pcb_defect_001.png --output-dir outputs/preprocessed_detection
```

当前 DeepPCB 测试记录（IoU=0.33）：mAP 0.3501、Precision 0.5580、Recall 0.4978、F1 0.5262。该结果用于课程项目流程验证，不代表工业检测性能。

## 目录结构

```text
pcb_florence2/
├── app.py                    # Gradio 前端
├── assets/                   # 页面资源
├── configs/                  # 训练配置
├── data/                     # 原始数据、JSONL 和划分
├── datasets/                 # 数据分类入口
├── weights/                  # adapter 和 checkpoints
├── training_results/         # 训练结果入口
├── scripts/                  # 检查、转换、预处理和推理
├── src/                      # 训练、评估、指标和报告模块
└── outputs/                  # 日志和可视化结果
```

## 说明与限制

- Florence-2 的核心 VLM 输出是原始生成序列中的缺陷类别和位置 token；数量统计、位置描述和中文报告由程序解析与规则生成。
- 当前置信度是生成序列分数映射得到的参考值，不等同于 YOLO 的校准目标级概率。
- 中文处理建议来自规则匹配，不是模型从 DeepPCB 学到的维修知识。
- 未检测到目标不等于 PCB 绝对合格，正式使用仍需人工和电气测试确认。
- DeepPCB 数据仅用于研究和课程项目，授权信息见 `data/raw/DeepPCB/LICENSE`。

## 参考项目

- DeepPCB：https://github.com/tangsanli5201/DeepPCB
- Florence-2 微调参考：https://github.com/microsoft/dstoolkit-finetuning-florence-2
- Florence-2 模型：https://huggingface.co/microsoft/Florence-2-base-ft
