import argparse,numpy as np
from control import Robot,ARM,HAND,save_result
p=argparse.ArgumentParser();p.add_argument('--motor',default='finger1_motor1',choices=ARM+HAND);p.add_argument('--mode',default='loss',choices=['loss','weak','jam']);p.add_argument('--all',action='store_true');p.add_argument('--viewer',action='store_true');a=p.parse_args();rows=[]
for name in (ARM+HAND if a.all else [a.motor]):
 for mode in (['loss','weak','jam'] if a.all else [a.mode]):
  r=Robot();r.step(.5);q0=float(r.data.joint(name).qpos[0]);r.fault(name,mode);r.command({name:q0+.3});maxforce=0.;maxmotion=0.
  for _ in range(100):
   r.step(.01);maxforce=max(maxforce,abs(float(r.data.actuator(name).force[0])));maxmotion=max(maxmotion,abs(float(r.data.joint(name).qpos[0])-q0))
  limit=float(max(abs(r.model.actuator(name).forcerange)))
  rows.append({'motor':name,'mode':mode,'max_force_Nm':maxforce,'torque_limit_Nm':limit,'max_drift_rad':maxmotion,'pass': bool(maxforce<1e-10 if mode=='loss' else maxmotion<.01 if mode=='jam' else maxforce<=limit+1e-9),'loop_error_m':r.loop_error()})
  r.restore(name)
save_result('faults.json',{'trials':rows,'all_pass':all(x['pass'] for x in rows)})
if a.viewer:
 import mujoco.viewer
 r.fault(a.motor,a.mode)
 with mujoco.viewer.launch_passive(r.model,r.data) as v:
  while v.is_running():r.step(.02,v)
