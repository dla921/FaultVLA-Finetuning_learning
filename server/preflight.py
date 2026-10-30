"""Check native splits, train-only statistics, sample alignment and optional GPU inference."""
import argparse,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['object','spatial'],default='object');p.add_argument('--inference',action='store_true');p.add_argument('--data-only',action='store_true');a=p.parse_args()
 import numpy as np
 from faultvla.stablegrasp import split_ids,separated_statistics as statistics,separated_samples as samples
 base=R/'data/SO101_StableGrasp200'
 counts={s:len(split_ids(base,s)) for s in ['train','validation','test']}
 stats=statistics(base);sample=next(samples(base,'train',1,1))
 assert sample['action'].shape==(5,7) and 'state' not in sample and len(stats['mean'])==7
 print('Native dataset/splits/train-only statistics OK:',counts,flush=True)
 if a.data_only:return
 import torch
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 GPU required'
 from faultvla.policy import validate_checkpoint
 c=json.loads((R/'configs/checkpoints.json').read_text())[a.model]
 validate_checkpoint(R/c['checkpoint'],c['unnorm_key'])
 if a.inference:
  from faultvla.stablegrasp_policy import Extractor
  feature=Extractor(R/c['checkpoint'],c['unnorm_key'])(sample['rgb'],sample['task'])
  assert feature.ndim==1 and torch.isfinite(feature).all()
  print('Single-camera frozen VLM inference OK:',list(feature.shape),flush=True)
if __name__=='__main__':main()
