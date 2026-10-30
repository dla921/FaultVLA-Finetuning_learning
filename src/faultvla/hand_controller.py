"""Sensor-free command mapping; no claim of measured position or contact force."""
import numpy as np

class HandController:
    # Source-data synergies: three identical opposing finger pairs plus thumb.
    @staticmethod
    def encode(targets):
        targets=np.asarray(targets,dtype=float)
        if targets.shape[-1:]!=(8,) or not np.isfinite(targets).all():
            raise ValueError('Expected finite eight-motor targets')
        commands=np.stack(((targets[...,0]+1)/1.5,targets[...,6]+1),axis=-1)
        reconstructed=HandController.targets(commands)
        if not np.allclose(reconstructed,targets,atol=1e-6):
            raise ValueError('Targets do not follow the source hand synergies')
        return commands

    @staticmethod
    def targets(commands):
        commands=np.asarray(commands,dtype=float)
        if commands.shape[-1:]!=(2,) or not np.isfinite(commands).all():
            raise ValueError('Expected finger_closure and thumb_closure')
        commands=np.clip(commands,0,1)
        fingers=-1+1.5*commands[...,0];thumb=-1+commands[...,1]
        return np.stack((fingers,-fingers,fingers,-fingers,fingers,-fingers,thumb,-thumb),axis=-1)
