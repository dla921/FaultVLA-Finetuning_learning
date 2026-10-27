"""Train final fc2 on all training episodes; never include test in model selection."""
import argparse,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--model',choices=['object','spatial'],default='object');p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--smoke',action='store_true');a=p.parse_args()
c=json.loads((R/'configs/checkpoints.json').read_text())[a.model]
cmd=[sys.executable,str(R/'scripts/train_limited.py'),'--checkpoint',str(R/c['checkpoint']),'--base-unnorm-key',c['unnorm_key'],'--data',str(R/'data/cartesian_demos_v2_wide200'),'--output',str(a.output.resolve()),'--epochs',str(a.epochs)]
cmd+=['--max-samples-per-episode','2'] if a.smoke else ['--all-episodes','--all-samples']
subprocess.run(cmd,check=True,cwd=R)
