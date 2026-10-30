"""Frozen camera-only VLM features, arm readout, hand commands and visual monitor.
The upstream 7D outputs are discarded, never interpreted as joint commands.
"""
import numpy as np
import torch
from faultvla.policy import Policy

class FeatureTap:
    def predict_action(self, hidden, **kwargs):
        self.feature = hidden[:, -1].float().mean(dim=1).detach().cpu()
        # Satisfy the upstream internal interface; these outputs are discarded.
        return torch.zeros((1,8,7),device=hidden.device,dtype=hidden.dtype)

class Extractor:
    def __init__(self, checkpoint, key):
        self.policy = Policy(checkpoint,key)
        p=self.policy
        p.cfg.use_proprio=False; p.cfg.num_images_in_input=1
        p.model.vision_backbone.set_num_images_in_input(1)
        p.model.requires_grad_(False)
        self.tap=FeatureTap()
    def __call__(self, rgb, task):
        p=self.policy
        with torch.inference_mode():
            p.infer(p.cfg,p.model,p.processor,{'full_image':rgb.copy()},task,
                    action_head=self.tap,proprio_projector=None,use_film=False,use_minivlm=True)
        return self.tap.feature.clone().squeeze(0)

CONTRACT = 'rgb-arm5-hand2-visual3-v1'

class SeparatedPolicy:
    def __init__(self,checkpoint,key,trained):
        payload=torch.load(trained,map_location='cpu',weights_only=True)
        if payload.get('contract')!=CONTRACT:raise ValueError('Incompatible checkpoint: retrain separated policy')
        self.stats=payload['statistics'];self.extract=Extractor(checkpoint,key)
        self.head=torch.nn.Linear(payload['input_dim'],38)
        self.head.load_state_dict(payload['head']);self.head.eval()
        self.arm_limits=np.asarray(payload['arm_limits'])

    def predict(self,rgb,task):
        from faultvla.hand_controller import HandController
        from faultvla.stablegrasp import VISUAL_STATES
        rgb=np.asarray(rgb)
        if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8:raise ValueError('Expected uint8 front RGB 224x224')
        if not task.strip():raise ValueError('Task instruction required')
        with torch.no_grad():
            feature=self.extract(rgb,task);self.last_camera_feature=feature.detach().cpu()
            output=self.head(feature)
        commands=output[:35].numpy().reshape(5,7)*np.array(self.stats['std'])+np.array(self.stats['mean'])
        if not np.isfinite(commands).all() or not torch.isfinite(output).all():raise ValueError('Nonfinite prediction')
        arm=np.clip(commands[:,:5],self.arm_limits[:,0],self.arm_limits[:,1])
        hand=np.clip(commands[:,5:],0,1)
        probabilities=torch.softmax(output[35:],dim=-1).numpy()
        return dict(raw_commands=commands,arm_targets_rad=arm,hand_commands=hand,hand_targets_rad=HandController.targets(hand),
                    visual_state=VISUAL_STATES[int(probabilities.argmax())],
                    visual_probabilities=dict(zip(VISUAL_STATES,probabilities.tolist())))


def load_policy(checkpoint,key,trained):
    payload=torch.load(trained,map_location='cpu',weights_only=True)
    if payload.get('contract')==CONTRACT:return SeparatedPolicy(checkpoint,key,trained)
    from faultvla.bridge_policy import BRIDGE_CONTRACT,BridgePolicy
    if payload.get('contract')==BRIDGE_CONTRACT:return BridgePolicy(checkpoint,key,trained)
    raise ValueError('Unknown policy contract')
