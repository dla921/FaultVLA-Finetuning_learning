"""RGB-feature hand stage classifier and command-only temporal handoff.
Source phases label training examples only; runtime uses camera features and
previously issued commands. No object pose, contacts, qpos or elapsed-task schedule.
"""
import numpy as np
import torch
from faultvla.hand_controller import HandController
CLASSES=('approach','ready_to_close','carry','release')
CONTRACT='camera-hand-stage-v1'
LABEL_SCHEME='preclose-close-only-v1'

def stage_labels(phases):
    labels=[];previous=0
    for phase in phases:
        phase=str(phase)
        if phase in ('02_preclose_settle','03_close'):previous=1
        elif phase in ('04_lift','04_grasp_check','05_transfer','06_place'):previous=2
        elif phase in ('07_release','08_retreat','09_open_thumb','Final'):previous=3
        elif phase=='settle':pass
        elif phase=='02_align' or phase.startswith(('00_','01_')):previous=0
        else:raise ValueError('Unknown source phase: '+phase)
        labels.append(previous)
    return np.array(labels,dtype=np.int64)

class StageHead(torch.nn.Module):
    def __init__(self,dim):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(dim,256),torch.nn.ReLU(),torch.nn.Linear(256,4))
    def forward(self,x):return self.layers(x)

class VisualHandJudge:
    def __init__(self,path,base_model):
        p=torch.load(path,map_location='cpu',weights_only=True)
        if p.get('contract')!=CONTRACT or p.get('label_scheme')!=LABEL_SCHEME or p['base_model']!=base_model:raise ValueError('Judge/base-model mismatch')
        self.mean=torch.tensor(p['mean']);self.std=torch.tensor(p['std'])
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or not (self.std>0).all():raise ValueError('Invalid feature normalization')
        self.head=StageHead(len(self.mean));self.head.load_state_dict(p['head']);self.head.eval()
    def predict(self,camera_feature):
        x=torch.as_tensor(camera_feature).detach().cpu().float()
        if x.shape!=self.mean.shape or not torch.isfinite(x).all():raise ValueError('Invalid camera feature')
        with torch.no_grad():p=torch.softmax(self.head((x-self.mean)/self.std),dim=-1).numpy()
        return p

class HandSupervisor:
    def __init__(self,threshold=.8,confirmations=3,speed=1.):
        if not np.isfinite(threshold) or not 0<threshold<=1 or confirmations<1 or not np.isfinite(speed) or speed<=0:raise ValueError('Invalid supervisor settings')
        self.threshold=threshold;self.confirmations=confirmations;self.speed=speed
        self.state='approach';self.ready_count=0;self.release_count=0;self.hold_frames=0;self.events=[]
    def update(self,probabilities,proposed_hand_commands,previous_issued_targets):
        p=np.asarray(probabilities,dtype=float);proposal=np.asarray(proposed_hand_commands,dtype=float)
        previous=np.asarray(previous_issued_targets,dtype=float).copy()
        if p.shape!=(4,) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):raise ValueError('Invalid stage probabilities')
        if proposal.shape!=(5,2) or not np.isfinite(proposal).all() or previous.shape!=(8,) or not np.isfinite(previous).all():raise ValueError('Invalid hand commands')
        event=None
        if self.state=='approach':
            self.ready_count=self.ready_count+1 if p[1]>=self.threshold else 0
            if self.ready_count>=self.confirmations:
                self.state='closing';event='close_command_triggered'
        elif self.state=='holding':
            self.hold_frames+=1
            self.release_count=self.release_count+1 if p[3]>=self.threshold else 0
            if self.hold_frames>=5 and self.release_count>=self.confirmations:
                self.state='released';event='release_command_triggered'
        if event:self.events.append(event)
        if self.state=='approach':
            # Preserve the baseline's approach aperture until camera-based takeover.
            targets=HandController.targets(proposal)
        else:
            goal=HandController.targets([1.,.7] if self.state in ('closing','holding') else [(.2/1.5),.2])
            targets=[]
            for _ in range(5):
                previous=previous+np.clip(goal-previous,-self.speed*.02,self.speed*.02)
                targets.append(previous.copy())
            targets=np.array(targets)
            if self.state=='closing' and np.max(np.abs(targets[-1]-goal))<1e-6:self.state='holding'
        return dict(targets=targets,state=self.state,event=event,ready_count=self.ready_count,
                    probabilities=dict(zip(CLASSES,p.tolist())))
