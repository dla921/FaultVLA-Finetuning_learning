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
 start=p.add_mutually_exclusive_group()
 start.add_argument('--start-phase',help='Diagnostic expert-prefix handoff at first source phase frame')
 start.add_argument('--start-index',type=int,help='Diagnostic expert-prefix handoff at this 50Hz index')
 p.add_argument('--backend',choices=['egl','glfw','osmesa'],default='egl')
 p.add_argument('--arm-max-speed',type=float,help='Optional arm command rate limit, rad/s')
 p.add_argument('--hand-max-speed',type=float,help='Optional hand command rate limit, rad/s')
 p.add_argument('--hand-judge',type=Path,help='Optional separately trained camera hand-stage classifier')
 p.add_argument('--judge-threshold',type=float,default=.8)
 p.add_argument('--judge-confirmations',type=int,default=3)
 p.add_argument('--judge-hand-speed',type=float,default=1.,help='Takeover hand command rate, rad/s')
 p.add_argument('--video',action='store_true');p.add_argument('--viewer',action='store_true');a=p.parse_args()
 if a.hand_judge and a.mode!='policy':p.error('--hand-judge requires policy mode')
 if a.mode=='policy' and not a.trained:p.error('--trained is required for policy mode')
 if a.steps is not None and a.steps<1:p.error('--steps must be positive')
 import math
 for rate in [a.arm_max_speed,a.hand_max_speed]:
  if rate is not None and (not math.isfinite(rate) or rate<=0):p.error('Command rate limits must be positive finite rad/s')
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
 from faultvla.rollout_start import select_start,replay_prefix
 with np.load(ep/'data.npz') as z:
  start_index=select_start(z['phase'],a.start_phase,a.start_index)
  start_phase=str(z['phase'][start_index])
  warmup_actions=z['action'][:start_index].copy() if start_index else None
  initial=z['full_qpos'][0].copy();velocity=z['qvel'][0].copy()
  reference=z['action'].copy() if a.mode=='replay' else None
  source=z['cup_xyz'][0].copy();target=z['cup_xyz'][-1].copy();total=len(z['time'])
 remaining=total-start_index
 steps=min(a.steps or remaining,remaining)
 remaining_complete=steps==remaining
 diagnostic_start=a.start_phase is not None or a.start_index is not None
 complete=not diagnostic_start and steps==total
 a.output.mkdir(parents=True,exist_ok=False)
 m=mujoco.MjModel.from_xml_path(str(root/'scene.xml'));d=mujoco.MjData(m)
 rm=mujoco.MjModel.from_xml_path(str(root/'scene_render.xml'));rd=mujoco.MjData(rm)
 if m.nq!=rm.nq or m.nu!=13:raise ValueError('Model dimensions differ')
 if [m.joint(i).name for i in range(m.njnt)]!=[rm.joint(i).name for i in range(rm.njnt)]:raise ValueError('Joint order differs')
 d.qpos[:]=initial;d.qvel[m.jnt_dofadr[m.actuator_trnid[:,0]]]=velocity
 d.ctrl[:]=d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]
 mujoco.mj_forward(m,d)
 if warmup_actions is not None:
  print('Diagnostic initialization: replaying',start_index,'source controls before handoff',flush=True)
  replay_prefix(m,d,warmup_actions)
  warmup_actions=None
 handoff_time=float(d.time)
 nsub=round(.02/m.opt.timestep)
 if not np.isclose(nsub*m.opt.timestep,.02):raise ValueError('Physics timestep mismatch')
 from faultvla.grasp_diagnostics import observe,DiagnosticSummary
 diagnostics=DiagnosticSummary();initial_diagnostics=observe(m,d)
 from faultvla.command_limiter import CommandLimiter
 limiter=CommandLimiter(d.ctrl,m.actuator_ctrlrange,arm_speed=a.arm_max_speed,hand_speed=a.hand_max_speed)
 previous=limiter.previous.copy();max_delta=np.zeros(13);limited_steps=0
 policy=None
 if a.mode=='policy':
  import torch
  from faultvla.stablegrasp_policy import load_policy
  payload=torch.load(a.trained,map_location='cpu',weights_only=True)
  c=json.loads((R/'configs/checkpoints.json').read_text())[payload['base_model']]
  policy=load_policy(R/c['checkpoint'],c['unnorm_key'],a.trained)
  if start_index and hasattr(policy,'observe_issued_command'):
   # These commands were actually executed during diagnostic initialization.
   with np.load(ep/'data.npz') as z:past_issued=z['action'][:start_index].copy()
   for issued in past_issued:policy.observe_issued_command(np.clip(issued,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]))
   del past_issued
 judge=supervisor=None;hand_events=[];hand_decision=None;initial_hand_probabilities=None
 if a.hand_judge:
  from faultvla.visual_hand import VisualHandJudge,HandSupervisor,CLASSES
  judge=VisualHandJudge(a.hand_judge,payload['base_model'])
  supervisor=HandSupervisor(a.judge_threshold,a.judge_confirmations,a.judge_hand_speed)
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
      hand_targets=HandController.targets(prediction['hand_commands'])
      if judge:
       probabilities=judge.predict(policy.last_camera_feature)
       if k==0:initial_hand_probabilities=dict(zip(CLASSES,probabilities.tolist()))
       hand_decision=supervisor.update(probabilities,prediction['hand_commands'],d.ctrl[5:].copy())
       hand_targets=hand_decision['targets']
       if hand_decision['event']:
        hand_events.append(dict(time=float(d.time),event=hand_decision['event']))
        print('Visual hand event',hand_events[-1],flush=True)
      chunk=np.concatenate((prediction['arm_targets_rad'],hand_targets),axis=1)
    action=chunk[k%5] if policy else reference[start_index+k]
    if not np.isfinite(action).all():raise ValueError('Nonfinite command')
    requested=np.clip(action,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    applied=limiter.apply(requested)
    limited_steps+=int(np.any(np.abs(applied-requested)>1e-12))
    max_delta=np.maximum(max_delta,np.abs(applied-previous));previous=applied.copy()
    d.ctrl[:]=applied
    if policy and hasattr(policy,'observe_issued_command'):policy.observe_issued_command(applied.copy())
    mujoco.mj_step(m,d,nstep=nsub)
    if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):raise RuntimeError('Invalid physics state/warning')
    mujoco.mj_forward(m,d)
    cup=d.body('demo_cube').xpos.copy();max_lift=max(max_lift,float(cup[2]-source[2]))
    record=dict(step=k,source_index=start_index+k,elapsed_after_handoff_s=float(d.time)-handoff_time,time=float(d.time),cup_xyz=cup.tolist(),command=d.ctrl.tolist(),requested_command=requested.tolist())
    telemetry=observe(m,d);diagnostics.update(d.time,telemetry);record['diagnostics']=telemetry
    if policy and k%5==0:
     record['visual_probabilities']=prediction['visual_probabilities']
     if 'command_context' in prediction:record['policy_command_context']=prediction['command_context']
    if judge and k%5==0:
     record['visual_hand_decision']={key:value for key,value in hand_decision.items() if key!='targets'}
     record['vla_hand_commands']=prediction['hand_commands'].tolist()
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
   uses_issued_command_context=bool(policy and hasattr(policy,'observe_issued_command')),
   steps=steps,complete_episode=complete,task_success=success if complete else None,
   initialization=dict(diagnostic=diagnostic_start,method='expert_prefix_physics_replay' if start_index else 'original_initial_state',
    source_start_index=start_index,source_start_phase=start_phase,expert_prefix_steps=start_index,
    handoff_time_s=handoff_time,initial_diagnostics=initial_diagnostics),
   completed_remaining_horizon=remaining_complete,
   diagnostic_remaining_task_success=success if diagnostic_start and remaining_complete else None,
   command_rate_limits_rad_s=dict(arm=a.arm_max_speed,hand=a.hand_max_speed),
   rate_limited_steps=limited_steps,max_command_delta_rad=max_delta.tolist(),
   visual_hand_supervisor=dict(enabled=bool(judge),checkpoint=str(a.hand_judge) if judge else None,
    final_state=supervisor.state if supervisor else None,events=hand_events,
    initial_probabilities=initial_hand_probabilities,
    threshold=a.judge_threshold if judge else None,confirmations=a.judge_confirmations if judge else None,
    takeover_hand_rate_rad_s=a.judge_hand_speed if judge else None),
   grasp_diagnostics=diagnostics.result(),
   max_lift_m=max_lift,final_cup_xyz=final.tolist(),target_reference_xyz=target.tolist(),
   upright_cos=upright,contacts=contacts,seconds=time.monotonic()-started,
   mujoco_version=mujoco.__version__,source_mujoco_version='3.3.7',render_backend=a.backend,
   note='Replay is source-action physics validation, not learned policy performance' if not policy else 'Policy uses RGB/task plus causal issued-command history when supported; no expert future actions or measured robot/object state after initialization')
  (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary,indent=2),flush=True)
 finally:
  if renderer:renderer.close()
  if writer:writer.close()
  if viewer:viewer.close()
 return 0 if not remaining_complete or success else 2
if __name__=='__main__':raise SystemExit(main())
