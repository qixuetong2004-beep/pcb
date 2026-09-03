#!/usr/bin/env python3
"""Resumable Florence-2 LoRA training; best adapter is selected by val mAP@0.33."""
import argparse,csv,json,math,random,shutil,time
from pathlib import Path
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForCausalLM,AutoProcessor,get_linear_schedule_with_warmup
from peft import LoraConfig,PeftModel,get_peft_model
from evaluate import load_rows,predict
from metrics import detection_metrics
class Rows(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return self.rows[i]
def collate(x):return x
def save_adapter(model,processor,path):
 path.mkdir(parents=True,exist_ok=True);model.save_pretrained(path);processor.save_pretrained(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/full_training.json');ap.add_argument('--resume',default='auto');ap.add_argument('--epochs',type=int);args=ap.parse_args()
 cfg=json.loads(Path(args.config).read_text());root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True);(root/'checkpoints').mkdir(exist_ok=True);(root/'training_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);device='cuda:0';dtype=torch.float16
 train_rows,val_rows=load_rows('data/splits/train.jsonl'),load_rows('data/splits/val.jsonl');loader=DataLoader(Rows(train_rows),batch_size=1,shuffle=True,num_workers=cfg['num_workers'],collate_fn=collate,pin_memory=True)
 checkpoints=sorted((root/'checkpoints').glob('epoch_*'));resume=Path(args.resume) if args.resume!='auto' else (checkpoints[-1] if checkpoints else None);processor=AutoProcessor.from_pretrained(cfg['model_id'],trust_remote_code=True)
 base=AutoModelForCausalLM.from_pretrained(cfg['model_id'],trust_remote_code=True,torch_dtype=dtype)
 if resume and (resume/'adapter_config.json').exists(): model=PeftModel.from_pretrained(base,resume,is_trainable=True);state=torch.load(resume/'trainer_state.pt',map_location='cpu');start=state['epoch'];best=state['best_map']
 else:
  if cfg['gradient_checkpointing']:
   try: base.gradient_checkpointing_enable();base.config.use_cache=False
   except Exception as e: print('gradient checkpointing unavailable:',e)
  lc=cfg['lora'];model=get_peft_model(base,LoraConfig(r=lc['r'],lora_alpha=lc['alpha'],lora_dropout=lc['dropout'],target_modules=lc['target_modules'],task_type='CAUSAL_LM',bias='none'));start=0;best=-1
 model.to(device);model.print_trainable_parameters();optim=AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);total=(len(loader)//cfg['gradient_accumulation_steps']+1)*(args.epochs or cfg['epochs']);sched=get_linear_schedule_with_warmup(optim,math.ceil(total*cfg['warmup_ratio']),total);scaler=torch.amp.GradScaler('cuda');history=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'logs').glob('epoch_*.json'))];torch.cuda.reset_peak_memory_stats()
 for epoch in range(start,args.epochs or cfg['epochs']):
  model.train();losses=[];begin=time.perf_counter();optim.zero_grad(set_to_none=True)
  for i,batch in enumerate(loader):
   row=batch[0];image=Image.open(row['image']).convert('RGB');inputs=processor(text=row['prefix'],images=image,return_tensors='pt').to(device,dtype);labels=processor.tokenizer(text=row['suffix'],return_tensors='pt',return_token_type_ids=False).input_ids.to(device)
   with torch.autocast('cuda',dtype=dtype): loss=model(input_ids=inputs['input_ids'],pixel_values=inputs['pixel_values'],labels=labels).loss/cfg['gradient_accumulation_steps']
   scaler.scale(loss).backward();losses.append(float(loss.detach())*cfg['gradient_accumulation_steps'])
   if (i+1)%cfg['gradient_accumulation_steps']==0 or i+1==len(loader): scaler.step(optim);scaler.update();optim.zero_grad(set_to_none=True);sched.step()
  model.eval();pred,timing=predict(model,processor,val_rows,cfg['max_new_tokens']);metrics=detection_metrics(pred);metrics.update(timing);metrics['epoch']=epoch+1;metrics['train_loss']=sum(losses)/len(losses);metrics['lr']=sched.get_last_lr()[0];metrics['seconds']=time.perf_counter()-begin;history.append(metrics)
  (root/'validation_predictions'/f'epoch_{epoch+1}.jsonl').parent.mkdir(exist_ok=True);(root/'validation_predictions'/f'epoch_{epoch+1}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pred));(root/'logs'/f'epoch_{epoch+1}.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
  ck=root/'checkpoints'/f'epoch_{epoch+1:02d}';save_adapter(model,processor,ck);torch.save({'epoch':epoch+1,'best_map':max(best,metrics['map'])},ck/'trainer_state.pt')
  if metrics['map']>best: best=metrics['map'];shutil.rmtree(root/'best_adapter',ignore_errors=True);shutil.copytree(ck,root/'best_adapter')
  old=sorted((root/'checkpoints').glob('epoch_*'))[:-cfg['keep_last_checkpoints']]
  for p in old: shutil.rmtree(p)
  print(json.dumps({'epoch':epoch+1,'loss':metrics['train_loss'],'val_map33':metrics['map'],'peak_mib':round(torch.cuda.max_memory_allocated()/1024**2,1)},ensure_ascii=False),flush=True)
 save_adapter(model,processor,root/'last_adapter');(root/'training_history.csv').write_text('')
 with (root/'training_history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['epoch','train_loss','lr','seconds','map','precision','recall','f1','valid_output_rate','mean_inference_seconds','peak_memory_mib'],extrasaction='ignore');w.writeheader();w.writerows(history)
 (root/'validation_metrics.json').write_text(json.dumps(history[-1],ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
