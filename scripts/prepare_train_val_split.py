#!/usr/bin/env python3
"""Create the fixed 900/100 training/validation split without touching test."""
import argparse, json, random
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--test', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--seed', type=int, default=42)
args=parser.parse_args()
records=[json.loads(x) for x in args.input.read_text(encoding='utf-8').splitlines() if x]
test=[json.loads(x) for x in args.test.read_text(encoding='utf-8').splitlines() if x]
if len(records)!=1000 or len(test)!=500: raise ValueError('Expected official 1000 trainval and 500 test records')
random.Random(args.seed).shuffle(records)
train, val=records[:900], records[900:]
paths=lambda rows: {x['image'] for x in rows}
if paths(train)&paths(val) or paths(train)&paths(test) or paths(val)&paths(test): raise RuntimeError('split leakage')
args.output_dir.mkdir(parents=True,exist_ok=True)
for name, rows in [('train',train),('val',val),('test',test)]:
    (args.output_dir/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
report={'seed':args.seed,'train':len(train),'val':len(val),'test':len(test),'overlap':0,'test_used_for_training_or_model_selection':False}
(args.output_dir/'split_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
