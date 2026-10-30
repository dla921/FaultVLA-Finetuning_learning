"""Causal context built only from issued targets, never measured robot state.
Stage names describe command progression, not contact or grasp success.
"""
import numpy as np
from faultvla.hand_controller import HandController
STAGES=('approach_command','closing_command','closed_command','opening_command')
CONTEXT_DIM=12
class CommandContext:
    def __init__(self,statistics):
        self.mean=np.asarray(statistics['mean'],dtype=float);self.std=np.asarray(statistics['std'],dtype=float)
        if self.mean.shape!=(7,) or self.std.shape!=(7,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0):raise ValueError('Invalid command statistics')
        self.previous=None;self.stage=0;self.has_closed=False
    def observe(self,issued):
        issued=np.asarray(issued,dtype=float)
        if issued.shape!=(13,) or not np.isfinite(issued).all():raise ValueError('Expected 13 issued targets')
        hand=HandController.encode(issued[5:]);command=np.concatenate((issued[:5],hand))
        main=float(issued[5]);thumb=float(issued[11])
        old_main=None if self.previous is None else float(-1+1.5*self.previous[5])
        if main>=.43 and abs(thumb+.3)<=.08:
            self.stage=2;self.has_closed=True
        elif self.has_closed and main<=-.65:
            self.stage=3
        elif self.has_closed and old_main is not None and main<old_main-1e-5:
            self.stage=3
        elif old_main is not None and main>old_main+1e-5:
            self.stage=1
        elif main<=-.65:
            self.stage=0
        self.previous=command
    def features(self):
        previous=np.zeros(7) if self.previous is None else (self.previous-self.mean)/self.std
        stage=np.eye(4)[self.stage]
        return np.concatenate((previous,stage,[float(self.previous is not None)])).astype(np.float32)
    def set_features(self,features):
        """Offline teacher-image evaluation only; restore causal source history."""
        features=np.asarray(features,dtype=float)
        if features.shape!=(CONTEXT_DIM,) or not np.isfinite(features).all():raise ValueError('Invalid context')
        if features[-1] not in (0,1) or not np.allclose(features[7:11],np.eye(4)[int(np.argmax(features[7:11]))]):raise ValueError('Invalid context stage/mask')
        self.previous=features[:7]*self.std+self.mean if features[-1] else None
        self.stage=int(np.argmax(features[7:11]));self.has_closed=self.stage in (2,3)

def contexts_from_actions(actions,statistics):
    context=CommandContext(statistics);result=[]
    for action in actions:
        result.append(context.features());context.observe(action)
    return np.stack(result)
