"""13-position-action interface in radians; passive joints are never commanded."""
from pathlib import Path
import json,time
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent
ARM=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
HAND=[f'finger{i}_motor{j}' for i in range(1,5) for j in (1,2)]
class Robot:
 def __init__(self):
  self.model=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));self.data=mujoco.MjData(self.model)
  h=np.load(ROOT/'home.npz');self.data.qpos[:]=h['qpos'];self.data.ctrl[:]=h['ctrl'];self.faults={};mujoco.mj_forward(self.model,self.data)
 def command(self,values):
  """Mapping {actuator_name: radians}; validates names and clips actuator limits."""
  for n,v in values.items():
   i=self.model.actuator(n).id
   if not np.isfinite(v):raise ValueError('Non-finite target')
   self.data.ctrl[i]=np.clip(v,*self.model.actuator_ctrlrange[i])
 def action(self,values):
  values=np.asarray(values)
  if values.shape!=(13,):raise ValueError('Expected 13 joint targets in radians')
  self.command(dict(zip(ARM+HAND,values)))
 def step(self,seconds=.02,viewer=None):
  for _ in range(round(seconds/self.model.opt.timestep)):
   mujoco.mj_step(self.model,self.data)
   if not np.all(np.isfinite(self.data.qpos)):raise RuntimeError('Non-finite simulation state')
   if any(w.number for w in self.data.warning):raise RuntimeError('MuJoCo warning: '+str(self.data.warning))
   if viewer is not None and viewer.is_running():viewer.sync();time.sleep(self.model.opt.timestep)
 def observe(self):
  return {'time':float(self.data.time),'qpos':{n:float(self.data.joint(n).qpos[0]) for n in ARM+HAND},'qvel':{n:float(self.data.joint(n).qvel[0]) for n in ARM+HAND},'cube_pos':self.data.body('demo_cube').xpos.copy(),'faults':dict(self.faults)}
 def fault(self,name,mode='loss',strength=.2):
  """loss: zero actuator gain AND bias; weak: scale torque limit; jam: physical soft equality lock."""
  if name in self.faults:raise ValueError('Restore before applying a new fault')
  i=self.model.actuator(name).id;j=self.model.actuator_trnid[i,0];e=self.model.equality('lock_'+name).id
  self.faults[name]=mode
  if not hasattr(self,'saved'):self.saved={}
  self.saved[name]=(self.model.actuator_gainprm[i].copy(),self.model.actuator_biasprm[i].copy(),self.model.actuator_forcerange[i].copy())
  if mode in ('loss','jam'):
   self.model.actuator_gainprm[i]=0;self.model.actuator_biasprm[i]=0
  elif mode=='weak':
   if not 0<=strength<=1:raise ValueError('strength must be in [0,1]')
   self.model.actuator_forcerange[i]*=strength
  else:raise ValueError(mode)
  if mode=='jam':self.model.eq_data[e,0]=self.data.qpos[self.model.jnt_qposadr[j]];self.data.eq_active[e]=1
 def restore(self,name):
  i=self.model.actuator(name).id
  self.model.actuator_gainprm[i],self.model.actuator_biasprm[i],self.model.actuator_forcerange[i]=self.saved.pop(name)
  self.data.eq_active[self.model.equality('lock_'+name).id]=0;self.faults.pop(name)
 def loop_error(self):
  e=[]
  for i in range(20):e.append(np.linalg.norm(self.data.site_xpos[self.model.eq_obj1id[i]]-self.data.site_xpos[self.model.eq_obj2id[i]]))
  return float(max(e))
 def rgbd(self,camera='front_rgbd',width=640,height=480):
  with mujoco.Renderer(self.model,height=height,width=width) as r:
   r.update_scene(self.data,camera=camera);rgb=r.render().copy();r.enable_depth_rendering();r.update_scene(self.data,camera=camera);depth=r.render().copy()
  return rgb,depth
 def cube_contacts(self):
  bid=self.model.body('demo_cube').id
  return [self.model.geom(c.geom2 if self.model.geom_bodyid[c.geom1]==bid else c.geom1).name for c in self.data.contact if (self.model.geom_bodyid[c.geom1]==bid) != (self.model.geom_bodyid[c.geom2]==bid)]

def save_result(name,result):
 (ROOT/'results').mkdir(exist_ok=True);(ROOT/'results'/name).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
