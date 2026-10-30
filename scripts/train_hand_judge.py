"""Fit a separate visual hand-stage classifier from normal demonstrations.
GPU VLM feature extraction; a small GPU classifier is trained with train-only
feature statistics. No physics or changes to the existing arm checkpoint.
"""
import argparse,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
import numpy as np,torch
from faultvla.stablegrasp import split_ids
from faultvla.stablegrasp_policy import Extractor
from faultvla.visual_hand import StageHead,stage_labels,CONTRACT,CLASSES,LABEL_SCHEME

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--model',choices=['object','spatial'],default='object');p.add_argument('--epochs',type=int,default=20)
 p.add_argument('--stride',type=int,default=5);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.epochs<1 or a.stride<1:p.error('Positive epochs/stride required')
 a.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(7);np.random.seed(7)
 from PIL import Image
 c=json.loads((R/'configs/checkpoints.json').read_text())[a.model];extract=Extractor(R/c['checkpoint'],c['unnorm_key'])
 root=R/'data/SO101_StableGrasp200';cache={};identities={}
 for split in ['train','validation']:
  features=[];labels=[];ids=[]
  for episode in split_ids(root,split)[:1 if a.smoke else None]:
   ep=root/'episodes'/episode;meta=json.loads((ep/'metadata.json').read_text())
   if not meta['accepted'] or not meta['images_saved']:raise ValueError('Invalid episode')
   with np.load(ep/'data.npz') as z:
    classes=stage_labels(z['phase']);indices=z['rgb_state_index'][::a.stride]
   if a.smoke:
    # One example from each stage for a wiring check, not quality validation.
    indices=np.array([indices[np.flatnonzero(classes[indices]==label)[0]] for label in range(4)])
   for index in indices:
    with Image.open(ep/'rgb'/f'{int(index):06d}.png') as image:rgb=np.array(image.convert('RGB'))
    features.append(extract(rgb,meta['instruction']));labels.append(int(classes[index]));ids.append([episode,int(index)])
    if len(features)%100==0:print(split,'features',len(features),flush=True)
  cache[split]=(torch.stack(features).float(),torch.tensor(labels,dtype=torch.long));identities[split]=ids
 x,y=cache['train'];mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-4)
 count=torch.bincount(y,minlength=4)
 if (count==0).any():raise ValueError('All hand stages required in train')
 head=StageHead(x.shape[1]).cuda();opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
 weight=(len(y)/(4*count.float())).cuda();best=float('inf');history=[];best_confusions=None;best_epoch=None
 initial=head.layers[0].weight.detach().clone()
 for epoch in range(1,a.epochs+1):
  head.train()
  for ids in torch.randperm(len(x)).split(64):
   opt.zero_grad();logits=head(((x[ids]-mean)/std).cuda())
   loss=torch.nn.functional.cross_entropy(logits,y[ids].cuda(),weight=weight)
   if not torch.isfinite(loss):raise ValueError('Nonfinite classifier loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
  head.eval();metrics={'epoch':epoch};confusions={}
  with torch.no_grad():
   for split,(features,labels) in cache.items():
    logits=head(((features-mean)/std).cuda());prediction=logits.argmax(-1).cpu()
    metrics[split]=float(torch.nn.functional.cross_entropy(logits,labels.cuda(),weight=weight))
    metrics[split+'_accuracy']=float((prediction==labels).float().mean())
    confusions[split]=torch.bincount(labels*4+prediction,minlength=16).reshape(4,4).tolist()
  history.append(metrics);print(json.dumps(metrics),flush=True)
  if metrics['validation']<best:
   best=metrics['validation'];best_confusions=confusions;best_epoch=epoch;payload=dict(contract=CONTRACT,label_scheme=LABEL_SCHEME,base_model=a.model,
    head={k:v.detach().cpu() for k,v in head.state_dict().items()},mean=mean.tolist(),std=std.tolist(),
    classes=list(CLASSES),epoch=epoch,stride=a.stride,smoke=a.smoke)
   temporary=a.output/'hand_judge.tmp';torch.save(payload,temporary);temporary.replace(a.output/'hand_judge.pt')
 assert not torch.equal(initial,head.layers[0].weight)
 assert all(p.grad is None for p in extract.policy.model.parameters())
 torch.save(dict(cache=cache,identities=identities),a.output/'features.pt')
 (a.output/'metrics.json').write_text(json.dumps(dict(history=history,best_epoch=best_epoch,best_confusion=best_confusions,last_confusion=confusions,classes=list(CLASSES),counts={key:len(value[1]) for key,value in cache.items()},smoke=a.smoke),indent=2))
 print('Saved camera-only hand judge; classifier updated, VLM frozen',flush=True)
if __name__=='__main__':main()
