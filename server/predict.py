"""Predict synchronized arm/hand commands from one front image and task text."""
import argparse,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--trained',type=Path,required=True)
 p.add_argument('--image',type=Path,required=True);p.add_argument('--task',required=True)
 p.add_argument('--issued-history',type=Path,help='Optional JSON list of previously issued 13D commands for coordinated policies')
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import numpy as np,torch
 from PIL import Image
 from faultvla.stablegrasp_policy import load_policy
 payload=torch.load(a.trained,map_location='cpu',weights_only=True)
 c=json.loads((R/'configs/checkpoints.json').read_text())[payload['base_model']]
 with Image.open(a.image) as image:rgb=np.array(image.convert('RGB'))
 policy=load_policy(R/c['checkpoint'],c['unnorm_key'],a.trained)
 if a.issued_history:
  if not hasattr(policy,'observe_issued_command'):raise ValueError('Issued history applies only to a coordinated policy')
  for command in json.loads(a.issued_history.read_text()):policy.observe_issued_command(command)
 result=policy.predict(rgb,a.task)
 result={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in result.items()}
 result.update(control_hz=50,camera_hz=10,contract=payload['contract'])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(result,f,indent=2)
 print('Saved camera-only arm/hand prediction:',a.output)
if __name__=='__main__':main()
