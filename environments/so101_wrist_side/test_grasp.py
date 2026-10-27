"""Contact-only scripted grasp trial. Failure is reported, never replaced with a weld."""
import argparse,json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from control import Robot,ROOT,ARM,HAND,save_result
p=argparse.ArgumentParser();p.add_argument('--lower',type=float,default=-.02);p.add_argument('--lift',type=float,default=.08);p.add_argument('--cube-offset',type=float,nargs=2,default=[-.10,0],metavar=('DX','DY'));p.add_argument('--viewer',action='store_true');p.add_argument('--capture',action='store_true');p.add_argument('--close',type=float,default=1.3);p.add_argument('--fault-motor',choices=ARM+HAND);p.add_argument('--fault-mode',default='loss',choices=['loss','weak','jam']);a=p.parse_args()
def ik(r,target,seed):
 m=r.model;d=mujoco.MjData(m);d.qpos[:]=r.data.qpos;ids=[m.joint(n).id for n in ARM];adr=m.jnt_qposadr[ids]
 def error(q):
  d.qpos[adr]=q;mujoco.mj_forward(m,d);R=d.body('r_wrist_interface').xmat.reshape(3,3)
  return np.r_[(d.site('palm_target').xpos-target)*10,(R[:,0]-[0,0,-1])*.3]
 fit=least_squares(error,seed,bounds=(m.jnt_range[ids,0]+.001,m.jnt_range[ids,1]-.001),max_nfev=250)
 error(fit.x);return fit.x,float(np.linalg.norm(d.site('palm_target').xpos-target))
def main():
 r=Robot();viewer=None
 if a.viewer:
  import mujoco.viewer
  viewer=mujoco.viewer.launch_passive(r.model,r.data)
 cube_adr=r.model.jnt_qposadr[r.model.body('demo_cube').jntadr[0]];r.data.qpos[cube_adr:cube_adr+2]+=a.cube_offset
 trace=[];r.step(.5,viewer);cube0=r.data.body('demo_cube').xpos.copy();q=np.array([r.data.joint(n).qpos[0] for n in ARM]);reach=[]
 def move(target,secs,label):
  start=r.data.ctrl.copy();target=np.array(target)
  for k in range(int(secs/.02)):
   u=(k+1)/int(secs/.02);u=u*u*(3-2*u);r.action(start+(target-start)*u);r.step(.02,viewer)
   trace.append({'time':float(r.data.time),'phase':label,'cube_z':float(r.data.body('demo_cube').xpos[2]),'contacts':r.cube_contacts(),'loop_error_m':r.loop_error()})
 for dz,label in [(.12,'approach'),(a.lower,'lower')]:
  q,err=ik(r,cube0+[0,0,dz],q);reach.append({'phase':label,'ik_error_m':err});target=r.data.ctrl.copy();target[:5]=q;move(target,2,label)
 # Coupled pair flexion pattern; tune amplitude/sign using recorded motor and tip response.
 target=r.data.ctrl.copy();target[5:]=np.tile([a.close,-a.close],4);move(target,2,'close');r.step(.5,viewer)
 if a.capture:
  from PIL import Image
  rgb,_=r.rgbd();Image.fromarray(rgb).save(ROOT/'results'/'grasp_closed.png')
 if a.fault_motor:r.fault(a.fault_motor,a.fault_mode)
 q,err=ik(r,cube0+[0,0,a.lift],q);reach.append({'phase':'lift','ik_error_m':err});target=r.data.ctrl.copy();target[:5]=q;move(target,2,'lift')
 hold=[]
 for _ in range(100):r.step(.02,viewer);hold.append(float(r.data.body('demo_cube').xpos[2]))
 if a.capture:
  from PIL import Image
  rgb,_=r.rgbd();Image.fromarray(rgb).save(ROOT/'results'/'grasp_lifted.png')
 gain=min(hold)-cube0[2];contacts=r.cube_contacts();success=bool(gain>.05 and any(n and n.startswith('contact_') for n in contacts))
 save_result('grasp.json',{'success':success,'lower_offset_m':a.lower,'lift_target_offset_m':a.lift,'cube_offset_xy_m':a.cube_offset,'hold_seconds':2.,'cube_initial_z_m':float(cube0[2]),'cube_final_z_m':float(r.data.body('demo_cube').xpos[2]),'held_lift_min_m':float(gain),'final_contacts':contacts,'ik':reach,'close_amplitude_rad':a.close,'fault':r.faults,'loop_error_m':r.loop_error(),'note':'No object weld, teleport during trial, or externally applied lifting force. False means grasp tuning required.','trace':trace})
 if viewer:viewer.close()
 return success
if __name__=='__main__':raise SystemExit(0 if main() else 2)
