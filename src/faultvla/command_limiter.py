"""Rate-limit target changes using previous commands, never measured joints."""
import numpy as np

class CommandLimiter:
    def __init__(self,initial,limits,dt=.02,arm_speed=None,hand_speed=None):
        self.limits=np.asarray(limits,dtype=float)
        initial=np.asarray(initial,dtype=float)
        if self.limits.shape!=(13,2) or initial.shape!=(13,):raise ValueError('Expected 13 actuators')
        if not np.isfinite(self.limits).all() or not np.isfinite(initial).all():raise ValueError('Nonfinite initial command/limits')
        if np.any(self.limits[:,0]>self.limits[:,1]):raise ValueError('Reversed limits')
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite timestep required')
        self.max_delta=np.full(13,np.inf)
        for speed,part in [(arm_speed,slice(0,5)),(hand_speed,slice(5,13))]:
            if speed is not None:
                if not np.isfinite(speed) or speed<=0:raise ValueError('Speed must be positive finite rad/s')
                self.max_delta[part]=speed*dt
        self.previous=np.clip(initial,self.limits[:,0],self.limits[:,1])

    def apply(self,target):
        target=np.asarray(target,dtype=float)
        if target.shape!=(13,) or not np.isfinite(target).all():raise ValueError('Expected finite 13D command')
        bounded=np.clip(target,self.limits[:,0],self.limits[:,1])
        applied=np.where(np.isinf(self.max_delta),bounded,
                         self.previous+np.clip(bounded-self.previous,-self.max_delta,self.max_delta))
        self.previous=applied.copy()
        return applied
