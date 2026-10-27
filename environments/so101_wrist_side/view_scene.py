import mujoco.viewer
from control import Robot
r=Robot()
with mujoco.viewer.launch_passive(r.model,r.data) as viewer:
 while viewer.is_running():r.step(.02,viewer)
