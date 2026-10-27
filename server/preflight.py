"""Check relocated data, statistics, GPU and an optional real checkpoint inference."""
import argparse,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['object','spatial'],default='object');p.add_argument('--inference',action='store_true');p.add_argument('--data-only',action='store_true');a=p.parse_args()
 import numpy as np
 from PIL import Image
 from faultvla.limited_training import load_split,training_statistics,examples
 base=R/'data/cartesian_demos_v2_wide200';counts={s:len(load_split(base,s)) for s in ['train','validation','test']}
 assert counts==dict(train=160,validation=20,test=20)
 stats,ids=training_statistics(base);assert len(ids)==160
 sample=next(examples(base,'train',maximum=1));row=sample['row'];ep=sample['directory']
 obs=dict(full_image=np.array(Image.open(ep/row['images']['front']).convert('RGB')),wrist_image=np.array(Image.open(ep/row['images']['wrist']).convert('RGB')),state=np.array(row['state'],dtype=np.float32))
 assert obs['full_image'].shape==obs['wrist_image'].shape==(224,224,3)
 print('Dataset/splits/train statistics OK:',counts,flush=True)
 if a.data_only:return
 import torch,mujoco
 from faultvla.policy import validate_checkpoint,Policy
 c=json.loads((R/'configs/checkpoints.json').read_text())[a.model]
 validate_checkpoint(R/c['checkpoint'],c['unnorm_key'])
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 GPU required'
 assert mujoco.__version__=='3.17.0','Recorded simulation requires MuJoCo 3.17.0'
 print('Checkpoint files and CUDA BF16 OK:',torch.cuda.get_device_name(),flush=True)
 if a.inference:
  policy=Policy(R/c['checkpoint'],c['unnorm_key'],replan_actions=8)
  result=policy.predict(obs,row['task']);assert result.shape==(8,7) and np.isfinite(result).all()
  print('Real checkpoint inference OK:',result.shape,flush=True)
if __name__=='__main__':main()
