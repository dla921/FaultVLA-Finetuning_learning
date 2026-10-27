"""Expert-only dataset/replay utilities. Never imported by run_vla.py."""
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from faultvla.bridge import Environment, ActionAdapter, ROOT, DEMO

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

def capture(robot):
    state=np.empty(mujoco.mj_stateSize(robot.model,SPEC))
    mujoco.mj_getState(robot.model,robot.data,state,SPEC)
    return state

def restore(robot,state):
    mujoco.mj_setState(robot.model,robot.data,state,SPEC)
    mujoco.mj_forward(robot.model,robot.data)

def configure_layout(robot, layout):
    # Episode reset only, before simulation/recording. Never teleport during task.
    robot.data.joint(robot.model.joint(robot.model.body('demo_cube').jntadr[0]).id).qpos[:3] = layout['cup_xyz']
    robot.model.geom('table2').pos[:2] = layout['table2_xy']
    mujoco.mj_forward(robot.model,robot.data)

def encode_target(adapter, target_position, target_rotation, open_value):
    p,r=adapter.pose()
    return np.r_[adapter.frame@(target_position-p)/adapter.cfg['translation_scale_m'],
        adapter.frame@Rotation.from_matrix(target_rotation@r.T).as_rotvec()/adapter.cfg['rotation_scale_rad'],open_value]

def target_pose(adapter, joint_targets):
    scratch=mujoco.MjData(adapter.m);scratch.qpos[:]=adapter.d.qpos
    scratch.qpos[adapter.qids[:5]]=joint_targets[:5]
    mujoco.mj_forward(adapter.m,scratch)
    return adapter.pose(scratch)

def evaluate(robot, initial_z, max_z, goal):
    pos=robot.data.body('demo_cube').xpos.copy()
    up=float(robot.data.body('demo_cube').xmat.reshape(3,3)[2,2]);contacts=robot.cube_contacts()
    success=bool(max_z-initial_z>.015 and abs(pos[0]-goal[0])<.03 and abs(pos[1]-goal[1])<.025
        and up>.95 and 'table2' in contacts and not any(c and c.startswith('contact_') for c in contacts))
    return dict(success=success,lift_m=max_z-initial_z,final_xyz=pos.tolist(),upright_cos=up,contacts=contacts)
