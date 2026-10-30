"""Train the upstream Pro Policy/Bridge Attention on RGB-only SO101 commands.
Frozen VLM, no LoRA. Online multilayer features avoid a terabyte-scale cache.
"""
import argparse,json,math,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
import numpy as np
import torch
from faultvla.stablegrasp import separated_samples,separated_statistics,split_ids
from faultvla.bridge_policy import BridgeExtractor,BridgeHead,BRIDGE_CONTRACT

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--model',choices=['object','spatial'],default='object')
 p.add_argument('--epochs',type=int,default=10);p.add_argument('--stride',type=int,default=5)
 p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--grad-accumulation',type=int,default=16)
 p.add_argument('--delta-loss-weight',type=float,default=.1);p.add_argument('--smoke',action='store_true')
 a=p.parse_args()
 if min(a.epochs,a.stride,a.grad_accumulation)<1:p.error('Positive epochs/stride/accumulation required')
 if not math.isfinite(a.learning_rate) or a.learning_rate<=0 or not math.isfinite(a.delta_loss_weight) or a.delta_loss_weight<0:p.error('Invalid loss/rate')
 a.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(7);np.random.seed(7)
 root=R/'data/SO101_StableGrasp200';stats=separated_statistics(root)
 c=json.loads((R/'configs/checkpoints.json').read_text())[a.model]
 extract=BridgeExtractor(R/c['checkpoint'],c['unnorm_key'])
 head=BridgeHead(extract.policy.head.model,extract.patches,reset_output=True).train()
 params=[p for p in head.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(params,lr=a.learning_rate,weight_decay=.01)
 max_ep=1 if a.smoke else None;max_samples=3 if a.smoke else None
 def records(split,shuffle=None):
  return separated_samples(root,split,max_ep,max_samples,stride=a.stride,evenly=a.smoke,shuffle_seed=shuffle)
 def count(split):
  total=0
  for name in split_ids(root,split)[:max_ep]:
   with np.load(root/'episodes'/name/'data.npz') as z:n=len(z['rgb_state_index'][::a.stride])
   total+=min(n,max_samples) if max_samples else n
  return total
 counts={s:count(s) for s in ['train','validation']}
 settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
 settings.update(contract=BRIDGE_CONTRACT,counts=counts,trainable_parameters=sum(p.numel() for p in params),vlm_frozen=True,lora=False)
 (a.output/'settings.json').write_text(json.dumps(settings,indent=2))
 steps_per_epoch=math.ceil(counts['train']/a.grad_accumulation);total_updates=steps_per_epoch*a.epochs
 warmup=max(1,round(total_updates*.1));updates=0
 def lr_factor(step):
  if step<warmup:return (step+1)/warmup
  return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_updates-warmup))))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
 checks={'query':head.network.mlp_resnet_blocks[0].q_proj.weight,
         'gate':head.network.mlp_resnet_blocks[0].gating_factor,'output':head.network.fc2.weight}
 before={key:param.detach().cpu().clone() for key,param in checks.items()}
 mean=torch.tensor(stats['mean'],device='cuda',dtype=torch.float32);std=torch.tensor(stats['std'],device='cuda',dtype=torch.float32)
 history=[];best=float('inf');started=time.monotonic()
 def loss_for(sample):
  hidden=extract(sample['rgb'],sample['task'])
  action,visual=head(hidden)
  target=(torch.tensor(sample['action'],device='cuda',dtype=torch.float32)-mean)/std
  l1=torch.nn.functional.l1_loss(action[0],target)
  delta=torch.nn.functional.l1_loss(action[0,1:]-action[0,:-1],target[1:]-target[:-1])
  ce=torch.nn.functional.cross_entropy(visual,torch.tensor([sample['visual_label']],device='cuda'))
  return l1+.1*ce+a.delta_loss_weight*delta
 for epoch in range(1,a.epochs+1):
  metric={'epoch':epoch}
  for split in ['train','validation']:
   head.train(split=='train');total=0.;seen=0;optimizer.zero_grad(set_to_none=True)
   for index,sample in enumerate(records(split,7+epoch if split=='train' else None)):
    with torch.set_grad_enabled(split=='train'):
     loss=loss_for(sample)
     if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
     if split=='train':
      window=min(a.grad_accumulation,counts[split]-(index//a.grad_accumulation)*a.grad_accumulation)
      (loss/window).backward()
      if (index+1)%a.grad_accumulation==0 or index+1==counts[split]:
       norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
       optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);updates+=1
    total+=float(loss.detach());seen+=1
    if seen%50==0 or seen==counts[split]:print(f'{split} epoch={epoch} samples={seen}/{counts[split]} loss={total/seen:.6f}',flush=True)
   if seen!=counts[split]:raise ValueError('Sample count mismatch')
   metric[split]=total/seen
  history.append(metric);print(json.dumps(metric),flush=True)
  changes={key:float((checks[key].detach().cpu()-before[key]).abs().max()) for key in checks}
  if not all(change>0 for change in changes.values()):raise RuntimeError('Bridge/query/output weights did not all update: '+str(changes))
  if any(param.grad is not None for param in extract.policy.model.parameters()):raise RuntimeError('Frozen VLM received gradients')
  if metric['validation']<best:
   best=metric['validation']
   import mujoco
   m=mujoco.MjModel.from_xml_path(str(root/'scene.xml'))
   payload=dict(contract=BRIDGE_CONTRACT,head={k:v.detach().cpu() for k,v in head.state_dict().items()},
                statistics=stats,base_model=a.model,arm_limits=m.actuator_ctrlrange[:5].tolist(),
                epoch=epoch,validation_loss=best,settings=settings,patches=extract.patches)
   temporary=a.output/'bridge_policy.tmp';torch.save(payload,temporary);temporary.replace(a.output/'bridge_policy.pt')
  (a.output/'metrics.json').write_text(json.dumps(dict(history=history,parameter_changes=changes,optimizer_updates=updates,seconds=time.monotonic()-started,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20),indent=2))
 print('Saved Bridge Attention policy; query/gate/output weights updated, VLM frozen',flush=True)
if __name__=='__main__':main()
