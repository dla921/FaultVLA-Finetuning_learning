import argparse,numpy as np
from control import Robot,ARM,HAND,save_result
p=argparse.ArgumentParser();p.add_argument('--viewer',action='store_true');a=p.parse_args()
def run(viewer=None):
 r=Robot();out=[]
 # Separate resets keep each motor trial comparable.
 for name in ARM+HAND:
  r=Robot();r.step(.4);q0=float(r.data.joint(name).qpos[0]);passive0=r.data.qpos.copy();target=q0+.12;r.command({name:target});r.step(1)
  delta=float(r.data.joint(name).qpos[0])-q0
  out.append({'motor':name,'target_rad':target,'motion_rad':delta,'loop_error_m':r.loop_error(),'moved':abs(delta)>.005})
 save_result('motors.json',{'nu':r.model.nu,'njnt':r.model.njnt,'nq':r.model.nq,'nv':r.model.nv,'loop_constraints':20,'trials':out,'all_moved':all(x['moved'] for x in out)})
 if a.viewer:
  import mujoco.viewer
  with mujoco.viewer.launch_passive(r.model,r.data) as v:
   while v.is_running():r.step(.02,v)
if __name__=='__main__':run()
