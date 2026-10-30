"""Camera-only adaptation of the upstream 24-block Pro policy, without LoRA.
All original multi-layer tokens are retained. No joint sensor input is fabricated.
"""
import numpy as np
import torch
from faultvla.stablegrasp_policy import Extractor
BRIDGE_CONTRACT='rgb-arm5-hand2-bridge24-visual3-v1'

class HiddenTap:
    def predict_action(self,hidden,**kwargs):
        self.hidden=hidden.detach()
        return torch.zeros((1,8,7),device=hidden.device,dtype=hidden.dtype)

class BridgeExtractor(Extractor):
    def __init__(self,checkpoint,key):
        super().__init__(checkpoint,key)
        self.tap=HiddenTap()
        self.patches=self.policy.model.vision_backbone.get_num_patches()
    def __call__(self,rgb,task):
        p=self.policy
        with torch.inference_mode():
            p.infer(p.cfg,p.model,p.processor,{'full_image':rgb.copy()},task,
                    action_head=self.tap,proprio_projector=None,use_film=False,use_minivlm=True)
        # Clone outside inference_mode so gradients through the policy are legal.
        hidden=self.tap.hidden.clone();self.tap.hidden=None
        return hidden

class BridgeHead(torch.nn.Module):
    def __init__(self,network,patches,reset_output=False):
        super().__init__();self.network=network.float();self.patches=patches
        dim=network.fc2.in_features
        self.null_proprio=torch.nn.Parameter(torch.zeros(1,1,dim,device=network.fc2.weight.device))
        self.visual=torch.nn.Linear(dim,3,device=network.fc2.weight.device)
        # Upstream FiLM parameters are present for compatibility but unused.
        for name,param in self.network.named_parameters():
            param.requires_grad_('film_gen' not in name)
        if reset_output:
            torch.nn.init.xavier_uniform_(self.network.fc2.weight)
            torch.nn.init.zeros_(self.network.fc2.bias)
    def conditioning(self,batch,context=None):
        return self.null_proprio.expand(batch,-1,-1)
    def forward(self,hidden,context=None):
        h=hidden.float();batch=h.shape[0];dim=self.network.fc2.in_features
        if h.shape[1]!=len(self.network.mlp_resnet_blocks)+1 or h.shape[2]<=self.patches:
            raise ValueError('Unexpected multi-layer token layout')
        x=torch.zeros((batch,8,7*dim),device=h.device,dtype=h.dtype)
        # Learned constant token replaces the sensor-conditioned upstream proprio token.
        action=self.network(x,h_a=h[:,:,self.patches:],h_t=h[:,:,:self.patches],
                            p=self.conditioning(batch,context))[:,:5]
        visual=self.visual(h[:,-1,:self.patches].mean(1))
        return action,visual

class BridgePolicy:
    def __init__(self,checkpoint,key,trained):
        payload=torch.load(trained,map_location='cpu',weights_only=True)
        if payload.get('contract')!=BRIDGE_CONTRACT:raise ValueError('Wrong Bridge policy checkpoint')
        self.extract=BridgeExtractor(checkpoint,key)
        self.head=BridgeHead(self.extract.policy.head.model,self.extract.patches)
        self.head.load_state_dict(payload['head']);self.head.eval()
        self.stats=payload['statistics'];self.arm_limits=np.asarray(payload['arm_limits'])
    def action_from_hidden(self,hidden):
        return self.head(hidden)
    def predict(self,rgb,task):
        from faultvla.hand_controller import HandController
        from faultvla.stablegrasp import VISUAL_STATES
        rgb=np.asarray(rgb)
        if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8 or not task.strip():raise ValueError('RGB/task required')
        with torch.no_grad():
            hidden=self.extract(rgb,task)
            self.last_camera_feature=hidden[:,-1].float().mean(1).detach().cpu().squeeze(0)
            action,visual=self.action_from_hidden(hidden)
        commands=action[0].cpu().numpy()*np.array(self.stats['std'])+np.array(self.stats['mean'])
        probabilities=torch.softmax(visual[0],dim=-1).cpu().numpy()
        if not np.isfinite(commands).all() or not np.isfinite(probabilities).all():raise ValueError('Nonfinite prediction')
        hand=np.clip(commands[:,5:],0,1)
        return dict(raw_commands=commands,arm_targets_rad=np.clip(commands[:,:5],self.arm_limits[:,0],self.arm_limits[:,1]),
                    hand_commands=hand,hand_targets_rad=HandController.targets(hand),
                    visual_state=VISUAL_STATES[int(probabilities.argmax())],
                    visual_probabilities=dict(zip(VISUAL_STATES,probabilities.tolist())))
