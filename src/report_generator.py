"""Conservative, rule-constrained Chinese PCB inspection reports."""
from collections import Counter

LABELS = {
    "open": ("开路", "高"), "open_circuit": ("开路", "高"),
    "short": ("短路", "高"), "short_circuit": ("短路", "高"),
    "mousebite": ("鼠咬", "中"), "mouse_bite": ("鼠咬", "中"),
    "spur": ("毛刺", "中"), "copper": ("多余铜", "中"),
    "spurious_copper": ("多余铜", "中"), "pin-hole": ("针孔", "中"),
    "pin_hole": ("针孔", "中"), "pinhole": ("针孔", "中"),
    "missing_hole": ("缺孔", "高"), "missing_pad": ("焊盘缺失", "高"),
}

SUGGESTIONS = {
    "开路": "建议进行局部放大检查，并结合导通测试确认线路连续性",
    "短路": "建议检查相邻铜线路，并进行绝缘性和电气连通性测试",
    "鼠咬": "建议检查线路是否变窄或接近断裂",
    "毛刺": "建议检查线路边缘和蚀刻加工质量",
    "多余铜": "建议检查相邻线路间距及异常导通风险",
    "针孔": "建议结合高分辨率原图进行显微复核",
    "缺孔": "建议复核对应孔位并进行工艺检查",
    "焊盘缺失": "建议复核焊盘区域并暂停直接使用该板",
}

def _region(box, size):
    x = ((box[0] + box[2]) / 2) / size[0]; y = ((box[1] + box[3]) / 2) / size[1]
    return ("左" if x < 1/3 else "右" if x > 2/3 else "中") + ("上" if y < 1/3 else "下" if y > 2/3 else "中") + "区域"

def generate_report(predictions, size, display_threshold=0.0, report_threshold=0.0):
    normalized = []
    for i, p in enumerate(predictions, 1):
        label = str(p.get("label", "")).lower(); info = LABELS.get(label)
        if not info: continue
        score = p.get("confidence"); score = float(score) if score is not None else 0.0
        item = dict(p); item.update(id=i, label_cn=info[0], risk=info[1], region=_region(p["bbox"], size), width=round(p["bbox"][2]-p["bbox"][0]), height=round(p["bbox"][3]-p["bbox"][1]), confidence=score)
        if score >= display_threshold: normalized.append(item)
    confirmed = [p for p in normalized if p["confidence"] >= report_threshold]
    suspected = [p for p in normalized if p["confidence"] < report_threshold]
    counts = Counter(p["label_cn"] for p in confirmed)
    if confirmed:
        summary = "、".join(f"{k}{v}处" for k, v in counts.items())
        positions = "、".join(sorted(set(p["region"] for p in confirmed)))
        text = f"本次检测共发现{len(confirmed)}处达到报告阈值的PCB缺陷，涉及{len(counts)}种缺陷类型，分别为{summary}。缺陷主要分布在{positions}。"
        high = any(p["risk"] == "高" for p in confirmed)
        text += "综合风险等级：高。建议暂停将该板直接投入使用，对标记区域进行人工复核，并结合导通、绝缘或显微检查确认。" if high else "综合风险等级：中。建议对标记区域进行放大检查，必要时进行返修或进一步电气测试。"
        for p in confirmed:
            text += f"编号{p['id']}为{p['label_cn']}，位于{p['region']}，坐标[{p['bbox'][0]}, {p['bbox'][1]}, {p['bbox'][2]}, {p['bbox'][3]}]，置信度{p['confidence']:.2f}；{SUGGESTIONS[p['label_cn']]}。"
    elif suspected:
        text = f"本次检测未发现达到正式报告阈值的确定缺陷，但检测到{len(suspected)}处低置信度疑似异常。建议对疑似区域进行局部放大检查，并结合原始彩色图像人工复核。"
    else:
        text = "本次检测未发现达到设定置信度阈值的明显PCB缺陷。建议结合原始高分辨率图像、人工目视检查及电气测试进行最终确认。"
    return text, normalized, {"confirmed": len(confirmed), "suspected": len(suspected), "display_threshold": display_threshold, "report_threshold": report_threshold}
