"""MuJoCo rollout for current StableGrasp200: replay baseline or RGB-only policy.
Original meshes drive physics; source render LOD supplies front RGB. No expert
state/action is used after initialization in policy mode. Object state is logged
only for evaluation. Short runs are connectivity checks, not task evaluations.
"""
import argparse,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=['replay','policy'],required=True)
 p.add_argument('--trained',type=Path);p.add_argument('--episode',help='Episode ID; defaults to the first test episode')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int)
 p.add_argument('--backend',choices=['egl','glfw','osmesa'],default='egl')
 p.add_argument('--video',action='store_true');p.add_argument('--viewer',action='store_true');a=p.parse_args()
 if a.mode=='policy' and not a.trained:p.error('--trained is required for policy mode')
 if a.steps is not None and a.steps<1:p.error('--steps must be positive')
 os.environ['MUJOCO_GL']=a.backend
 import numpy as np,mujoco
 from PIL import Image
 from faultvla.hand_controller import HandController
 from faultvla.stablegrasp import split_ids
 root=R/'data/SO101_StableGrasp200'
 splits={name:split_ids(root,name) for name in ['train','validation','test']}
 if a.episode is None:a.episode=splits['test'][0]
 if a.episode not in sum(splits.values(),[]):raise ValueError('Unknown episode')
 split=next(name for name,ids in splits.items() if a.episode in ids)
 ep=root/'episodes'/a.episode;meta=json.loads((ep/'metadata.json').read_text())
 with np.load(ep/'data.npz') as z:
  initial=z['full_qpos'][0].copy();velocity=z['qvel'][0].copy()
  reference=z['action'].copy() if a.mode=='replay' else None
  source=z['cup_xyz'][0].copy();target=z['cup_xyz'][-1].copy();total=len(z['time'])
 steps=min(a.steps or total,total);complete=steps==total
 a.output.mkdir(parents=True,exist_ok=False)
 m=mujoco.MjModel.from_xml_path(str(root/'scene.xml'));d=mujoco.MjData(m)
 rm=mujoco.MjModel.from_xml_path(str(root/'scene_render.xml'));rd=mujoco.MjData(rm)
 if m.nq!=rm.nq or m.nu!=13:raise ValueError('Model dimensions differ')
 if [m.joint(i).name for i in range(m.njnt)]!=[rm.joint(i).name for i in range(rm.njnt)]:raise ValueError('Joint order differs')
 d.qpos[:]=initial;d.qvel[m.jnt_dofadr[m.actuator_trnid[:,0]]]=velocity
 d.ctrl[:]=d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]
 mujoco.mj_forward(m,d)
 nsub=round(.02/m.opt.timestep)
 if not np.isclose(nsub*m.opt.timestep,.02):raise ValueError('Physics timestep mismatch')
 policy=None
 if a.mode=='policy':
  import torch
  from faultvla.stablegrasp_policy import SeparatedPolicy
  payload=torch.load(a.trained,map_location='cpu',weights_only=True)
  c=json.loads((R/'configs/checkpoints.json').read_text())[payload['base_model']]
  policy=SeparatedPolicy(R/c['checkpoint'],c['unnorm_key'],a.trained)
 viewer=writer=renderer=None
 def rgb():
  rd.qpos[:]=d.qpos;mujoco.mj_kinematics(rm,rd);mujoco.mj_comPos(rm,rd);mujoco.mj_camlight(rm,rd)
  renderer.update_scene(rd,camera='front_rgbd')
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
  return renderer.render().copy()
 started=time.monotonic();max_lift=0.;chunk=None
 try:
  renderer=mujoco.Renderer(rm,height=224,width=224)
  if a.viewer:
   import mujoco.viewer
   viewer=mujoco.viewer.launch_passive(m,d)
  if a.video:
   import imageio.v2 as imageio
   writer=imageio.get_writer(str(a.output/'rollout.gif'),mode='I',duration=100,loop=0)
  with (a.output/'trace.jsonl').open('w') as log:
   for k in range(steps):
    if k%5==0:
     frame=rgb()
     if k==0:Image.fromarray(frame).save(a.output/'first.png')
     if writer:writer.append_data(frame)
     if policy:
      prediction=policy.predict(frame,meta['instruction'])
      chunk=np.concatenate((prediction['arm_targets_rad'],HandController.targets(prediction['hand_commands'])),axis=1)
    action=chunk[k%5] if policy else reference[k]
    if not np.isfinite(action).all():raise ValueError('Nonfinite command')
    d.ctrl[:]=np.clip(action,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    mujoco.mj_step(m,d,nstep=nsub)
    if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):raise RuntimeError('Invalid physics state/warning')
    mujoco.mj_forward(m,d)
    cup=d.body('demo_cube').xpos.copy();max_lift=max(max_lift,float(cup[2]-source[2]))
    record=dict(step=k,time=float(d.time),cup_xyz=cup.tolist(),command=d.ctrl.tolist())
    if policy and k%5==0:record['visual_probabilities']=prediction['visual_probabilities']
    log.write(json.dumps(record)+'\n')
    if viewer:
     if not viewer.is_running():raise RuntimeError('Viewer closed before completion')
     viewer.sync();time.sleep(.02)
    if (k+1)%100==0:print('Physics steps',k+1,'/',steps,flush=True)
   final_frame=rgb();Image.fromarray(final_frame).save(a.output/'last.png')
   if writer:writer.append_data(final_frame)
  bid=m.body('demo_cube').id
  contacts=[m.geom(c.geom2 if m.geom_bodyid[c.geom1]==bid else c.geom1).name for c in d.contact
            if (m.geom_bodyid[c.geom1]==bid)!=(m.geom_bodyid[c.geom2]==bid)]
  final=d.body('demo_cube').xpos.copy();upright=float(d.body('demo_cube').xmat.reshape(3,3)[2,2])
  success=bool(max_lift>.015 and np.linalg.norm(final[:2]-target[:2])<.03 and upright>.95
               and 'table2' in contacts and not any(n and n.startswith('contact_') for n in contacts))
  summary=dict(mode=a.mode,episode=a.episode,split=split,trained=str(a.trained) if a.trained else None,
   steps=steps,complete_episode=complete,task_success=success if complete else None,
   max_lift_m=max_lift,final_cup_xyz=final.tolist(),target_reference_xyz=target.tolist(),
   upright_cos=upright,contacts=contacts,seconds=time.monotonic()-started,
   mujoco_version=mujoco.__version__,source_mujoco_version='3.3.7',render_backend=a.backend,
   note='Replay is source-action physics validation, not learned policy performance' if not policy else 'Policy uses RGB/task only; no expert actions or states after initialization')
  (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary,indent=2),flush=True)
 finally:
  if renderer:renderer.close()
  if writer:writer.close()
  if viewer:viewer.close()
 return 0 if not complete or success else 2
if __name__=='__main__':raise SystemExit(main())
