"""Frozen Object-Pro visual/language features plus a native joint readout.
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

class JointPolicy:
    def __init__(self, checkpoint, key, trained):
        payload=torch.load(trained,map_location='cpu',weights_only=True)
        self.stats=payload['statistics']; self.extract=Extractor(checkpoint,key)
        self.head=torch.nn.Linear(payload['input_dim'],65)
        self.head.load_state_dict(payload['head']);self.head.eval()
    def predict(self,rgb,qpos,task):
        qpos=np.asarray(qpos,dtype=np.float32)
        if qpos.shape!=(13,) or not np.isfinite(qpos).all(): raise ValueError('Expected 13 joint positions')
        s=self.stats['qpos'];state=(qpos-np.array(s['mean']))/np.array(s['std'])
        x=torch.cat((self.extract(rgb,task),torch.tensor(state,dtype=torch.float32)))
        with torch.no_grad(): normalized=self.head(x).numpy().reshape(5,13)
        s=self.stats['action'];result=normalized*np.array(s['std'])+np.array(s['mean'])
        if not np.isfinite(result).all(): raise ValueError('Nonfinite joint prediction')
        return result
