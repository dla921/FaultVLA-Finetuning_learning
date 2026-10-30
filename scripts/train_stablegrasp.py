"""Train a 5x13 joint readout on frozen single-camera VLM features."""
import argparse,copy,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
import numpy as np
import torch
from faultvla.stablegrasp import samples,statistics
from faultvla.stablegrasp_policy import Extractor

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--smoke',action='store_true')
    p.add_argument('--model',choices=['object','spatial'],default='object');a=p.parse_args()
    if a.epochs<1: raise ValueError('Positive epochs required')
    torch.manual_seed(7);np.random.seed(7)
    a.output.mkdir(parents=True,exist_ok=False)
    root=R/'data/SO101_StableGrasp200';stats=statistics(root)
    c=json.loads((R/'configs/checkpoints.json').read_text())[a.model]
    extract=Extractor(R/c['checkpoint'],c['unnorm_key'])
    cache={}
    for split in ['train','validation']:
        xs=[];ys=[];identities=[]
        for s in samples(root,split,1 if a.smoke else None,2 if a.smoke else None):
            norm=lambda x,key:(x-np.array(stats[key]['mean']))/np.array(stats[key]['std'])
            feature=extract(s['rgb'],s['task'])
            xs.append(torch.cat((feature,torch.tensor(norm(s['state'],'qpos'),dtype=torch.float32))))
            ys.append(torch.tensor(norm(s['action'],'action').reshape(-1),dtype=torch.float32))
            identities.append(dict(episode=s['episode'],index=s['index']))
            if len(xs)%100==0: print(split,len(xs),flush=True)
        cache[split]=(torch.stack(xs),torch.stack(ys))
        (a.output/(split+'_samples.json')).write_text(json.dumps(identities))
    head=torch.nn.Linear(cache['train'][0].shape[1],65)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=0)
    initial=head.weight.detach().clone();history=[];best=None;best_loss=float('inf')
    for epoch in range(1,a.epochs+1):
        head.train();x,y=cache['train']
        for ids in torch.randperm(len(x)).split(64):
            optimizer.zero_grad();loss=torch.nn.functional.l1_loss(head(x[ids]),y[ids])
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);optimizer.step()
        head.eval();metrics={'epoch':epoch}
        with torch.no_grad():
            for split,(x,y) in cache.items():metrics[split]=float(torch.nn.functional.l1_loss(head(x),y))
        history.append(metrics);print(json.dumps(metrics),flush=True)
        if metrics['validation']<best_loss:best_loss=metrics['validation'];best=copy.deepcopy(head.state_dict())
    assert not torch.equal(initial,head.weight)
    assert all(p.grad is None for p in extract.policy.model.parameters())
    payload=dict(head=best,input_dim=head.in_features,statistics=stats,base_model=a.model,
                 action_shape=[5,13],control_hz=50,rgb_hz=10)
    torch.save(payload,a.output/'joint_readout.pt')
    (a.output/'metrics.json').write_text(json.dumps(dict(history=history,smoke=a.smoke,statistics_episodes=160),indent=2))
    from faultvla.stablegrasp_policy import JointPolicy
    policy=JointPolicy.__new__(JointPolicy);policy.stats=stats;policy.extract=extract;policy.head=head
    head.load_state_dict(torch.load(a.output/'joint_readout.pt',weights_only=True)['head'])
    s=next(samples(root,'validation',1,1));prediction=policy.predict(s['rgb'],s['state'],s['task'])
    assert prediction.shape==(5,13) and np.isfinite(prediction).all()
    print('Saved native joint readout; online validation prediction OK',flush=True)
if __name__=='__main__': main()
