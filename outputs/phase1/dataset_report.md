# DeepPCB 数据检查报告

- 数据根目录：`/home/qiyue/图像处理/DeepPCB/PCBData`
- 官方划分：trainval 1000，test 500，图像交集 0
- 已引用：1500 张测试图、1500 张模板图、1500 个标注；发现 1500 个测试 JPG、1501 个模板 JPG、1500 个标注 TXT
- 图像尺寸：640x640 × 1500
- 缺陷框总数：10013

## 类别框数量

- open（1）：1942
- short（2）：1506
- mousebite（3）：1965
- spur（4）：1625
- copper（5）：1474
- pin-hole（6）：1501

## 完整性检查

- missing_images：0
- missing_templates：0
- missing_annotations：0
- parse_errors：0
- invalid_boxes：0
- empty_annotations：0

## 可视化样本

- `/home/qiyue/图像处理/pcb_florence2/outputs/phase1/visualizations/annotation_example_01_12100122_test.jpg`
- `/home/qiyue/图像处理/pcb_florence2/outputs/phase1/visualizations/annotation_example_02_20085228_test.jpg`
- `/home/qiyue/图像处理/pcb_florence2/outputs/phase1/visualizations/annotation_example_03_20085051_test.jpg`
- `/home/qiyue/图像处理/pcb_florence2/outputs/phase1/visualizations/annotation_example_04_00041074_test.jpg`
- `/home/qiyue/图像处理/pcb_florence2/outputs/phase1/visualizations/annotation_example_05_00041012_test.jpg`
- `/home/qiyue/图像处理/pcb_florence2/outputs/phase1/visualizations/annotation_example_06_13000166_test.jpg`
