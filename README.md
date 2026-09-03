# Florence-2 PCB 缺陷检测与智能描述

基于 `microsoft/Florence-2-base-ft` 的 PCB 生成式目标检测项目。模型接受 PCB 图像与 `<OD>` 提示词，生成缺陷类别及位置 token，再解析为边界框和中文检测报告。

## 已训练结果

- 最佳 LoRA adapter：`outputs/full_training/best_adapter/`（验证集第 11 轮）
- 官方 DeepPCB 测试集：mAP@IoU 0.33 为 0.3501，Precision 为 0.5580，Recall 为 0.4978，F1 为 0.5262。
- 平均推理时间：0.151 秒/图；峰值显存：693.7 MiB。

## 快速启动

```bash
conda activate pcb
cd pcb_florence2
python app.py
```

访问 `http://127.0.0.1:7860`，上传 `data/raw/DeepPCB/PCBData/**/**_test.jpg`。

## 项目内容

项目根目录提供了便于查看的分类入口（使用符号链接指向实际目录，不会重复占用磁盘）：

- `weights/`：最佳 adapter、最后一轮 adapter 和 checkpoints。
- `datasets/`：DeepPCB、Mendeley 处理图像和 Florence-2 JSONL 记录。
- `training_results/`：完整训练输出和训练配置。
- `scripts/`、`src/`：数据检查、预处理、训练、评估和报告代码。
- `assets/`：界面资源图片。

- `data/raw/DeepPCB/`：DeepPCB 原始仓库的数据工作树（不包含上游 Git 历史）。
- `data/processed/`：Florence-2 `<OD>` JSONL 转换结果。
- `data/splits/`：固定随机种子 42 的 900/100/500 训练、验证、测试划分。
- `outputs/full_training/best_adapter/`：最佳 LoRA 权重。
- `outputs/full_training/test_predictions/`：官方测试集预测、指标和可视化。
- `app.py`：展示 VLM 提示词、原始生成序列、结构化检测结果及中文报告的 Gradio 页面。

## 数据与模型说明

DeepPCB 数据仅供研究用途，详见 `data/raw/DeepPCB/LICENSE` 及上游仓库：https://github.com/tangsanli5201/DeepPCB 。

本仓库保存的是 LoRA adapter，不包含 Florence-2 基座权重；首次运行会从 Hugging Face 加载 `microsoft/Florence-2-base-ft`。
