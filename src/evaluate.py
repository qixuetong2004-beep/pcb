#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoProcessor
from metrics import CLASSES,detection_metrics,gt_from_suffix

def load_rows(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]
def predict(model,processor,rows,max_new_tokens=512,visual_dir=None):
    device='cuda:0'; output=[]; elapsed=[]; torch.cuda.reset_peak_memory_stats()
    for n,row in enumerate(rows):
        image=Image.open(row['image']).convert('RGB'); inp=processor(text='<OD>',images=image,return_tensors='pt').to(device,torch.float16)
        begin=time.perf_counter()
        with torch.inference_mode():
            generated=model.generate(input_ids=inp['input_ids'],pixel_values=inp['pixel_values'],max_new_tokens=max_new_tokens,num_beams=3,do_sample=False,return_dict_in_generate=True,output_scores=True)
        elapsed.append(time.perf_counter()-begin); text=processor.batch_decode(generated.sequences,skip_special_tokens=False)[0]
        try:
            parsed=processor.post_process_generation(text,task='<OD>',image_size=image.size)['<OD>']; labels=parsed.get('labels',[]); boxes=parsed.get('bboxes',[])
            confidence=float(generated.sequences_scores[0]) if getattr(generated,'sequences_scores',None) is not None else 0.0
            preds=[{'label':l,'bbox':[round(x) for x in b],'confidence':confidence} for l,b in zip(labels,boxes) if l in CLASSES]; valid=True
        except Exception: preds=[]; valid=False
        item={'image':row['image'],'ground_truth':gt_from_suffix(row['suffix'],*image.size),'predictions':preds,'valid':valid,'raw_sequence':text}; output.append(item)
        if visual_dir and n<30:
            d=ImageDraw.Draw(image)
            for p in preds: d.rectangle(p['bbox'],outline='red',width=2); d.text((p['bbox'][0],p['bbox'][1]),p['label'],fill='red')
            image.save(Path(visual_dir)/f'{n:03d}_{Path(row["image"]).name}')
    return output,{'mean_inference_seconds':sum(elapsed)/len(elapsed),'peak_memory_mib':round(torch.cuda.max_memory_allocated()/1024**2,1)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--adapter',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--max-new-tokens',type=int,default=512);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/'visualizations').mkdir(exist_ok=True)
 processor=AutoProcessor.from_pretrained(a.adapter,trust_remote_code=True);base=AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base-ft',trust_remote_code=True,torch_dtype=torch.float16);model=PeftModel.from_pretrained(base,a.adapter).to('cuda:0').eval(); records,timing=predict(model,processor,load_rows(a.index),a.max_new_tokens,out/'visualizations'); metrics=detection_metrics(records);metrics.update(timing);(out/'predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));(out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n');print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
