"""Pro Policy conditioned on causal issued-command history and command stage."""
import numpy as np
import torch
from faultvla.bridge_policy import BridgeHead,BridgeExtractor,BridgePolicy
from faultvla.command_context import CommandContext,CONTEXT_DIM
CONTRACT='rgb-arm5-hand2-bridge24-context12-v1'
class CoordinatedHead(BridgeHead):
    def __init__(self,network,patches,reset_output=False):
        super().__init__(network,patches,reset_output)
        dim=network.fc2.in_features;device=network.fc2.weight.device
        self.context_projection=torch.nn.Sequential(torch.nn.Linear(CONTEXT_DIM,256),torch.nn.ReLU(),torch.nn.Linear(256,dim)).to(device)
    def conditioning(self,batch,context=None):
        if context is None or context.shape!=(batch,CONTEXT_DIM) or not torch.isfinite(context).all():raise ValueError('Causal command context required')
        return self.null_proprio.expand(batch,-1,-1)+self.context_projection(context.float()).unsqueeze(1)
class CoordinatedPolicy(BridgePolicy):
    def __init__(self,checkpoint,key,trained):
        payload=torch.load(trained,map_location='cpu',weights_only=True)
        if payload.get('contract')!=CONTRACT:raise ValueError('Wrong coordinated checkpoint')
        self.extract=BridgeExtractor(checkpoint,key)
        self.head=CoordinatedHead(self.extract.policy.head.model,self.extract.patches)
        self.head.load_state_dict(payload['head']);self.head.eval()
        self.stats=payload['statistics'];self.arm_limits=np.asarray(payload['arm_limits'])
        self.command_context=CommandContext(self.stats)
    def reset_command_context(self):self.command_context=CommandContext(self.stats)
    def observe_issued_command(self,issued):self.command_context.observe(issued)
    def set_offline_context(self,features):self.command_context.set_features(features)
    def action_from_hidden(self,hidden):
        context=torch.tensor(self.command_context.features(),device=hidden.device).unsqueeze(0)
        return self.head(hidden,context)
    def predict(self,rgb,task):
        features=self.command_context.features().tolist()
        result=super().predict(rgb,task);result['command_context']=features
        return result
