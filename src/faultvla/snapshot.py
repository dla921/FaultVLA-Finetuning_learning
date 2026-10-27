"""Full MuJoCo state snapshots for same-model continuation."""
import hashlib
from pathlib import Path
import numpy as np
import mujoco

def signature(model):
    # Dimensions and names catch accidental use of a different robot layout.
    return hashlib.sha256(bytes(model.names) + str((model.nq,model.nv,model.nu,model.neq)).encode()).hexdigest()

def save(path, robot, initial_z, max_z, completed_steps):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(robot.model, spec))
    mujoco.mj_getState(robot.model, robot.data, state, spec)
    path = Path(path)
    temp = path.with_suffix('.tmp.npz')
    np.savez(temp, state=state, signature=signature(robot.model),
             mujoco_version=mujoco.__version__, initial_z=initial_z,
             max_z=max_z, completed_steps=completed_steps)
    temp.replace(path)

def restore(path, robot):
    with np.load(path, allow_pickle=False) as snap:
        if str(snap['signature']) != signature(robot.model) or str(snap['mujoco_version']) != mujoco.__version__:
            raise ValueError('Snapshot model layout or MuJoCo version mismatch')
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        if snap['state'].shape != (mujoco.mj_stateSize(robot.model, spec),):
            raise ValueError('Snapshot state size mismatch')
        mujoco.mj_setState(robot.model, robot.data, snap['state'], spec)
        mujoco.mj_forward(robot.model, robot.data)
        return {k: snap[k].item() for k in ['initial_z','max_z','completed_steps']}
