import argparse,json,numpy as np,mujoco
from scipy.optimize import least_squares
from control import Robot,ROOT,save_result
p=argparse.ArgumentParser();p.add_argument('--capture',action='store_true');p.add_argument('--radial',type=float,default=0.);p.add_argument('--close',type=float,default=.5);p.add_argument('--offset',type=float,default=0.);p.add_argument('--video',action='store_true');p.add_argument('--viewer',action='store_true');a=p.parse_args()
r=Robot();m=r.model;d=r.data;viewer=None
if a.viewer:
 from mujoco import viewer as mv
 viewer=mv.launch_passive(m,d)
q=np.array([0,-.5,-.5,-.5,0.]);log=[];writer=renderer=None
if a.video or a.capture:
 import imageio.v2 as imageio
 renderer=mujoco.Renderer(m,height=480,width=640)
 if a.video:writer=imageio.get_writer(str(ROOT/'results'/'side_transfer.mp4'),fps=5)
def frame(label):
 if renderer:
  renderer.update_scene(d,camera='front_rgbd');im=renderer.render().copy()
  from PIL import Image,ImageDraw
  im=Image.fromarray(im);ImageDraw.Draw(im).text((15,15),"WRIST-ROLL SIDE GRASP: "+label,fill=(20,20,20));
  if writer:writer.append_data(np.array(im))
  return im

def ik(target):
 global q
 dd=mujoco.MjData(m);dd.qpos[:]=d.qpos;n=np.array([target[0]-.0388353,target[1],0]);n/=np.linalg.norm(n)
 def f(qq):
  dd.qpos[:5]=qq;mujoco.mj_forward(m,dd);R=dd.body('r_wrist_interface').xmat.reshape(3,3);return np.r_[(dd.site('palm_target').xpos-target)*10,(R[:,0]-np.cross([0,0,1],n))*2,(R[:,2]-n)*2]
 fits=[least_squares(f,seed,bounds=(m.jnt_range[:5,0]+.001,m.jnt_range[:5,1]-.001),max_nfev=300) for seed in [q,[0,-.5,-.5,-.5,0],[0,-1,.5,-1,0]]];fit=min(fits,key=lambda x:np.linalg.norm(x.fun));q=fit.x;f(q);return q.copy(),float(np.linalg.norm(dd.site('palm_target').xpos-target))

def move(target,close,sec,label):
 qq,err=ik(target);cmd=d.ctrl.copy();cmd[:5]=qq;cmd[5:]=np.tile([close,-close],4)
 if close>0:cmd[11:]=[1.5,-1.5]
 start=d.ctrl.copy()
 for k in range(round(sec/.02)):
  t=(k+1)/round(sec/.02);t=t*t*(3-2*t);r.action(start+(cmd-start)*t);r.step(.02,viewer)
  if writer and k%10==0:frame(label)
 log.append({'phase':label,'ik_error':err,'palm_normal_world':d.body('r_wrist_interface').xmat.reshape(3,3)[:,0].tolist(),'wrist_roll_rad':float(d.joint('wrist_roll').qpos[0]),'cup_xyz':d.body('demo_cube').xpos.tolist(),'upright':float(d.body('demo_cube').xmat.reshape(3,3)[2,2])})
 if renderer:frame(label).save(ROOT/'results'/(label+'.png'))

cup=np.array([.26,-.08,.12]);n=np.array([cup[0]-.0388353,cup[1],0]);n/=np.linalg.norm(n);n=np.cross([0,0,1],n)
# Initial robot placement is performed before the demonstration starts, without moving the cup.
qq,err=ik(cup-n*.04+[0,0,.02]);d.qpos[:5]=qq;d.ctrl[:5]=qq;mujoco.mj_forward(m,d);r.step(.5,viewer);cup0=d.body('demo_cube').xpos.copy()
move(cup0-n*.035+[0,0,a.offset+.005],-.7,2,'01_side_approach');move(cup0+n*a.radial+[0,0,a.offset],-.7,2,'02_align');move(cup0+n*a.radial+[0,0,a.offset],a.close,2,'03_close');r.step(.5,viewer)
move(cup0+[0,0,.03+a.offset],a.close,2,'04_lift');r.step(.5,viewer);lift=float(d.body('demo_cube').xpos[2]-cup0[2]);relative=d.body('demo_cube').xpos-d.site('palm_target').xpos
# Rotate cup-to-palm displacement with palm yaw to avoid changing the grasp frame.
R0=d.body('r_wrist_interface').xmat.reshape(3,3).copy();local=R0.T@relative
for pos,label in [(np.array([.28,.05,cup0[2]+.03]),'05_transfer'),(np.array([.28,.05,cup0[2]+.002]),'06_place')]:
 n2=np.array([pos[0]-.0388353,pos[1],0]);n2/=np.linalg.norm(n2);R=np.column_stack([np.cross([0,0,1],n2),[0,0,1],n2]);target=pos-R@local;move(target,a.close,4,label)
move(d.site('palm_target').xpos.copy(),-.8,2,'07_release');r.step(.5,viewer)
n2=np.array([.26-.0388353,.08,0]);n2/=np.linalg.norm(n2);move(d.site('palm_target').xpos.copy()-np.cross([0,0,1],n2)*.09+np.array([0,0,.05]),-.8,2,'08_retreat');r.step(2.,viewer)
pos=d.body('demo_cube').xpos.copy();up=float(d.body('demo_cube').xmat.reshape(3,3)[2,2]);ct=r.cube_contacts();success=bool(lift>.015 and abs(pos[0]-.26)<.03 and abs(pos[1]-.08)<.025 and up>.95 and 'table2' in ct and not any(c and c.startswith('contact_') for c in ct))
save_result('side_transfer.json',{'success':success,'lift_m':lift,'final_xyz':pos.tolist(),'upright_cos':up,'contacts':ct,'stages':log})
if renderer:frame('Final').save(ROOT/'results'/'final.png')
if writer:writer.close()
if renderer:renderer.close()
if viewer:viewer.close()
raise SystemExit(0 if success else 2)
