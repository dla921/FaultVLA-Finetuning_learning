"""Predict synchronized arm/hand commands from one front image and task text."""
import argparse,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--trained',type=Path,required=True)
 p.add_argument('--image',type=Path,required=True);p.add_argument('--task',required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import numpy as np,torch
 from PIL import Image
 from faultvla.stablegrasp_policy import SeparatedPolicy,CONTRACT
 payload=torch.load(a.trained,map_location='cpu',weights_only=True)
 if payload.get('contract')!=CONTRACT:raise ValueError('Retrain using current separated workflow')
 c=json.loads((R/'configs/checkpoints.json').read_text())[payload['base_model']]
 with Image.open(a.image) as image:rgb=np.array(image.convert('RGB'))
 policy=SeparatedPolicy(R/c['checkpoint'],c['unnorm_key'],a.trained)
 result=policy.predict(rgb,a.task)
 result={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in result.items()}
 result.update(control_hz=50,camera_hz=10,contract=CONTRACT)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(result,f,indent=2)
 print('Saved camera-only arm/hand prediction:',a.output)
if __name__=='__main__':main()
