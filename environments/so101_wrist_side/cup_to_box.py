"""Contact-only scripted grasp trial. Failure is reported, never replaced with a weld."""
import argparse,json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from control import Robot,ROOT,ARM,HAND,save_result
p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');p.add_argument('--lower',type=float,default=-.02);p.add_argument('--lift',type=float,default=.08);p.add_argument('--cube-offset',type=float,nargs=2,default=[-.10,0],metavar=('DX','DY'));p.add_argument('--viewer',action='store_true');p.add_argument('--capture',action='store_true');p.add_argument('--close',type=float,default=1.3);p.add_argument('--fault-motor',choices=ARM+HAND);p.add_argument('--fault-mode',default='loss',choices=['loss','weak','jam']);a=p.parse_args()
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
  from mujoco import viewer as mjviewer
  viewer=mjviewer.launch_passive(r.model,r.data)
 cube_adr=r.model.jnt_qposadr[r.model.body('demo_cube').jntadr[0]];r.data.qpos[cube_adr:cube_adr+2]+=a.cube_offset
 writer=None;renderer=None
 if a.video:
  import imageio.v2 as imageio
  writer=imageio.get_writer(str(ROOT/'results'/'cup_to_box.mp4'),fps=10);renderer=mujoco.Renderer(r.model,height=480,width=640)
 trace=[];r.step(.5,viewer);cube0=r.data.body('demo_cube').xpos.copy();q=np.array([r.data.joint(n).qpos[0] for n in ARM]);reach=[]
 def move(target,secs,label):
  start=r.data.ctrl.copy();target=np.array(target)
  for k in range(int(secs/.02)):
   u=(k+1)/int(secs/.02);u=u*u*(3-2*u);r.action(start+(target-start)*u);r.step(.02,viewer)
   if writer and k%5==0:
    renderer.update_scene(r.data,camera='front_rgbd');writer.append_data(renderer.render())
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
 r.step(1.,viewer)
 if a.capture:
  from PIL import Image
  rgb,_=r.rgbd();Image.fromarray(rgb).save(ROOT/'results'/'cup_lifted.png')
 lifted=float(r.data.body('demo_cube').xpos[2]-cube0[2]);handpos=r.data.site('palm_target').xpos.copy();obj=r.data.body('demo_cube').xpos.copy();relative=obj-handpos
 for target_cup,label in [(np.array([.24,.105,obj[2]]),'transport'),(np.array([.24,.105,.095]),'place')]:
  q,err=ik(r,target_cup-relative,q);reach.append({'phase':label,'ik_error_m':err});target=r.data.ctrl.copy();target[:5]=q;move(target,3,label)
 target=r.data.ctrl.copy();target[5:]=0;move(target,2,'release');r.step(2.,viewer)
 retreat=r.data.site('palm_target').xpos.copy()+[-.02,0,.12];q,err=ik(r,retreat,q);target=r.data.ctrl.copy();target[:5]=q;move(target,3,'retreat');r.step(2.,viewer)
 contacts=r.cube_contacts()
 pos=r.data.body('demo_cube').xpos.copy();success=bool(lifted>.05 and abs(pos[0]-.24)<.03 and abs(pos[1]-.105)<.03 and .07<pos[2]<.13 and 'box_floor' in contacts and not any(n and n.startswith('contact_') for n in contacts) and np.linalg.norm(r.data.body('demo_cube').cvel)<.05)
 if a.capture:
  from PIL import Image
  rgb,_=r.rgbd();Image.fromarray(rgb).save(ROOT/'results'/'cup_in_box.png')
 save_result('cup_to_box.json',{'success':success,'lift_m':lifted,'final_cup_pos':pos.tolist(),'contacts':r.cube_contacts(),'ik':reach,'loop_error_m':r.loop_error(),'trace':trace})
 if writer:writer.close();renderer.close()
 if viewer:viewer.close()
 return success
if __name__=='__main__':raise SystemExit(0 if main() else 2)
