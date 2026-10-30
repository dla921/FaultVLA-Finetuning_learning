"""Diagnostic handoff selection and expert-prefix physics initialization only."""
import numpy as np

def select_start(phases,start_phase=None,start_index=None):
    if start_phase is not None and start_index is not None:raise ValueError('Choose phase or index')
    if start_phase is not None:
        matches=np.flatnonzero(np.asarray(phases)==start_phase)
        if not len(matches):raise ValueError('Source phase not found: '+start_phase)
        return int(matches[0])
    index=0 if start_index is None else start_index
    if not isinstance(index,(int,np.integer)) or not 0<=index<len(phases):raise ValueError('Start index outside episode')
    return int(index)

def replay_prefix(model,data,actions):
    """Restore dynamics via source commands before handoff; no policy is running."""
    import mujoco
    actions=np.asarray(actions)
    if actions.ndim!=2 or actions.shape[1]!=model.nu or not np.isfinite(actions).all():raise ValueError('Invalid prefix actions')
    nsub=round(.02/model.opt.timestep)
    if nsub<1 or not np.isclose(nsub*model.opt.timestep,.02):raise ValueError('Physics timestep mismatch')
    for action in actions:
        data.ctrl[:]=np.clip(action,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
        mujoco.mj_step(model,data,nstep=nsub)
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):raise RuntimeError('Prefix physics failure')
        mujoco.mj_forward(model,data)
