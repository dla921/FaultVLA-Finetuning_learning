"""Train camera-only arm/hand command readout and a visual location monitor."""
import argparse,copy,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
import numpy as np
import torch
from faultvla.stablegrasp import separated_samples as samples,separated_statistics as statistics
from faultvla.stablegrasp_policy import Extractor,CONTRACT

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
        xs=[];ys=[];labels=[];identities=[]
        for s in samples(root,split,1 if a.smoke else None,2 if a.smoke else None):
            norm=lambda x:(x-np.array(stats['mean']))/np.array(stats['std'])
            feature=extract(s['rgb'],s['task'])
            xs.append(feature)
            ys.append(torch.tensor(norm(s['action']).reshape(-1),dtype=torch.float32))
            labels.append(s['visual_label'])
            identities.append(dict(episode=s['episode'],index=s['index']))
            if len(xs)%100==0: print(split,len(xs),flush=True)
        cache[split]=(torch.stack(xs),torch.stack(ys),torch.tensor(labels,dtype=torch.long))
        (a.output/(split+'_samples.json')).write_text(json.dumps(identities))
    head=torch.nn.Linear(cache['train'][0].shape[1],38)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=0)
    initial=head.weight.detach().clone();history=[];best=None;best_loss=float('inf')
    for epoch in range(1,a.epochs+1):
        head.train();x,y,labels=cache['train']
        for ids in torch.randperm(len(x)).split(64):
            optimizer.zero_grad();out=head(x[ids])
            loss=torch.nn.functional.l1_loss(out[:,:35],y[ids])+.1*torch.nn.functional.cross_entropy(out[:,35:],labels[ids])
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);optimizer.step()
        head.eval();metrics={'epoch':epoch}
        with torch.no_grad():
            for split,(x,y,labels) in cache.items():
                out=head(x);metrics[split]=float(torch.nn.functional.l1_loss(out[:,:35],y)+.1*torch.nn.functional.cross_entropy(out[:,35:],labels))
                metrics[split+'_visual_accuracy']=float((out[:,35:].argmax(-1)==labels).float().mean())
        history.append(metrics);print(json.dumps(metrics),flush=True)
        if metrics['validation']<best_loss:best_loss=metrics['validation'];best=copy.deepcopy(head.state_dict())
    assert not torch.equal(initial,head.weight)
    assert all(p.grad is None for p in extract.policy.model.parameters())
    # Resolve actuator control ranges from the source model without feeding state to policy.
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(root/'scene.xml'))
    payload=dict(head=best,input_dim=head.in_features,statistics=stats,base_model=a.model,
                 contract=CONTRACT,arm_limits=model.actuator_ctrlrange[:5].tolist(),
                 action_shape=[5,7],control_hz=50,rgb_hz=10)
    torch.save(payload,a.output/'separated_policy.pt')
    (a.output/'metrics.json').write_text(json.dumps(dict(history=history,smoke=a.smoke,statistics_episodes=160,contract=CONTRACT),indent=2))
    from faultvla.stablegrasp_policy import SeparatedPolicy
    policy=SeparatedPolicy.__new__(SeparatedPolicy);policy.stats=stats;policy.extract=extract;policy.head=head
    policy.arm_limits=np.array(payload['arm_limits'])
    head.load_state_dict(torch.load(a.output/'separated_policy.pt',weights_only=True)['head'])
    sample=next(samples(root,'validation',1,1));prediction=policy.predict(sample['rgb'],sample['task'])
    assert prediction['arm_targets_rad'].shape==(5,5) and prediction['hand_targets_rad'].shape==(5,8)
    print('Saved camera-only separated policy; arm/hand/vision online prediction OK',flush=True)
if __name__=='__main__': main()
