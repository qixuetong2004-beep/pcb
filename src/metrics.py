"""Florence-2 detection parsing and dependency-free detection metrics."""
import re
from collections import defaultdict

CLASSES=("open","short","mousebite","spur","copper","pin-hole")
TOKEN=re.compile(r"(open|short|mousebite|spur|copper|pin-hole)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
def gt_from_suffix(s, w=640, h=640):
    return [{"label":m.group(1),"bbox":[round(int(m.group(i))/999*(w if i in(2,4) else h)) for i in range(2,6)]} for m in TOKEN.finditer(s)]
def iou(a,b):
    """IoU for either raw coordinate sequences or detection dictionaries."""
    a = a["bbox"] if isinstance(a, dict) else a
    b = b["bbox"] if isinstance(b, dict) else b
    if len(a) != 4 or len(b) != 4:
        raise ValueError("IoU requires [x1, y1, x2, y2] boxes")
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.0
def _ap(points):
    # Precision envelope / integral AP.
    r=[0.0]+[x[0] for x in points]+[1.0]; p=[0.0]+[x[1] for x in points]+[0.0]
    for i in range(len(p)-2,-1,-1): p[i]=max(p[i],p[i+1])
    return sum((r[i]-r[i-1])*p[i] for i in range(1,len(r)))
def detection_metrics(records, threshold=0.33):
    result={}; aps=[]; total_tp=total_fp=total_fn=0
    for label in CLASSES:
        gts=defaultdict(list); preds=[]
        for r in records:
            gts[r['image']]+=[x for x in r['ground_truth'] if x['label']==label]
            preds += [(x.get('confidence',0.0),r['image'],x) for x in r['predictions'] if x['label']==label]
        preds.sort(reverse=True,key=lambda x:x[0]); used=defaultdict(set); tp=[]; fp=[]
        for _,image,pred in preds:
            candidates=[(iou(pred['bbox'],g),idx) for idx,g in enumerate(gts[image]) if idx not in used[image]]
            best=max(candidates,default=(0,-1))
            if best[0]>=threshold: used[image].add(best[1]); tp.append(1); fp.append(0)
            else: tp.append(0); fp.append(1)
        n=sum(map(len,gts.values())); ctp=0; cfp=0; points=[]
        for a,b in zip(tp,fp): ctp+=a; cfp+=b; points.append((ctp/n if n else 0,ctp/(ctp+cfp)))
        ctp=sum(tp); cfp=sum(fp); cfn=n-ctp; prec=ctp/(ctp+cfp) if ctp+cfp else 0; rec=ctp/n if n else 0
        ap=_ap(points); result[label]={"precision":prec,"recall":rec,"f1":2*prec*rec/(prec+rec) if prec+rec else 0,"ap":ap,"gt":n,"pred":len(preds)}
        aps.append(ap); total_tp+=ctp; total_fp+=cfp; total_fn+=cfn
    precision=total_tp/(total_tp+total_fp) if total_tp+total_fp else 0; recall=total_tp/(total_tp+total_fn) if total_tp+total_fn else 0
    return {"iou_threshold":threshold,"classes":result,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0,"map":sum(aps)/len(aps),"valid_output_rate":sum(bool(r['valid']) for r in records)/len(records) if records else 0}
