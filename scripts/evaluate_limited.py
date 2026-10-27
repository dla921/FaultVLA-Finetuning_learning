"""Evaluate a limited-training checkpoint with real closed-loop VLA actions.
Only the episode's initial state is restored. No demonstration actions are read.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'VLA-Adapter')]
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--episode',default='validation_0000')
    p.add_argument('--data',type=Path,default=ROOT/'data/cartesian_demos_v2_wide200')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=470)
    p.add_argument('--render-backend', choices=['egl','glfw','dataset'], default='egl')
    p.add_argument('--viewer',action='store_true')
    args=p.parse_args()
    if not 1 <= args.steps <= 470:
        raise ValueError('Use 1..470 steps; fewer than 470 is a partial diagnostic')
    manifest_early=json.loads((args.data/'manifest.json').read_text())
    entry_early=next((e for e in manifest_early['episodes'] if e['id']==args.episode),None)
    if entry_early is None:raise ValueError('Unknown dataset episode')
    rendering_early=json.loads((args.data/entry_early['path']/'rendering.json').read_text())
    if args.render_backend=='dataset' and rendering_early.get('backend')=='glfw/d3d12':
        os.environ['MUJOCO_GL']='glfw'
        os.environ['GALLIUM_DRIVER']='d3d12'
        os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME']='NVIDIA'
    else:
        os.environ['MUJOCO_GL']='glfw' if args.viewer else ('egl' if args.render_backend=='dataset' else args.render_backend)
    import numpy as np
    import mujoco
    from PIL import Image
    from faultvla.bridge import Environment,ActionAdapter
    from faultvla.expert_data import configure_layout,restore,evaluate
    from faultvla.policy import Policy
    from faultvla.snapshot import save
    cfg=json.loads((args.training_run/'inference_config.json').read_text())
    manifest=json.loads((args.data/'manifest.json').read_text())
    allowed={entry['id']:entry for entry in manifest['episodes']}
    if args.episode not in allowed:raise ValueError('Unknown dataset episode')
    ep=args.data/allowed[args.episode]['path']
    meta=json.loads((ep/'metadata.json').read_text())
    profile=json.loads((ep/'rendering.json').read_text())
    # State vectors are valid only in the original MuJoCo model/version.
    if (ep/'quality.json').exists():
        recorded_version=json.loads((ep/'quality.json').read_text())['mujoco_version']
    else:
        source=ROOT/meta['source_episode']
        recorded_version=json.loads((source/'metadata.json').read_text())['mujoco_version']
    if recorded_version!=mujoco.__version__:raise ValueError('MuJoCo version mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    robot=Environment(args.output/'model');configure_layout(robot,meta['layout'])
    with np.load(ep/'trajectory.npz') as recorded:
        restore(robot,recorded['states'][0])
    robot.model.vis.quality.offsamples=profile['multisamples']
    robot.model.vis.quality.shadowsize=256
    adapter=ActionAdapter(robot,cfg)
    policy=Policy(args.training_run.resolve()/'checkpoint',cfg['unnorm_key'],replan_actions=1)
    viewer=None
    if args.viewer:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(robot.model,robot.data)
    initial=float(robot.data.body('demo_cube').xpos[2]);highest=initial
    summary=dict(mode='real_vla_limited_checkpoint',episode=args.episode,
        split=allowed[args.episode]['split'],checkpoint=cfg['checkpoint'],requested_steps=args.steps,
        completed_steps=0,rejected=0,clipped=0,error=None,task_success=None,
        initialization='demonstration initial state only; no trajectory actions',rendering=profile,
        evaluation_render_backend=os.environ['MUJOCO_GL'], pixel_identical_to_dataset=False)
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2))
    images=args.output/'rgb';images.mkdir()
    try:
        with mujoco.Renderer(robot.model,height=224,width=224) as renderer, (args.output/'actions.jsonl').open('w') as log:
            for step in range(args.steps):
                obs={'state':adapter.state()}
                for key,camera,label in [('full_image','front_rgbd','front'),('wrist_image','vla_wrist','wrist')]:
                    renderer.update_scene(robot.data,camera=camera)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=profile['shadows']
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=profile['reflections']
                    obs[key]=renderer.render().copy()
                    Image.fromarray(obs[key]).save(images/f'{step:05d}_{label}.png')
                action=policy.predict(obs,cfg['task'])[0]
                command,report=adapter.execute(action,cfg['control_dt'],viewer)
                highest=max(highest,float(robot.data.body('demo_cube').xpos[2]))
                summary['completed_steps']+=1
                summary['rejected']+=not report['accepted'];summary['clipped']+=report['action_clipped']
                log.write(json.dumps(dict(step=step,state=obs['state'].tolist(),action=action.tolist(),
                    motor_targets_rad=command.tolist(),cup_xyz=robot.data.body('demo_cube').xpos.tolist(),**report))+'\n');log.flush()
                save(args.output/'last_good.npz',robot,initial,highest,summary['completed_steps'])
                if step==0 or (step+1)%50==0:print(f'VLA control: {step+1}/{args.steps}',flush=True)
        result=evaluate(robot,initial,highest,meta['layout']['goal_xy'])
        summary['final_metrics']=result
        if args.steps==470:summary['task_success']=result['success']
    except Exception as exc:
        summary['error']=repr(exc)
        if hasattr(adapter,'last_stop'):summary['speed_stop']=adapter.last_stop
    finally:
        if viewer:viewer.close()
        summary['full_horizon_completed']=summary['completed_steps']==470
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2),flush=True)
    if summary['error']:return 1
    return 2 if summary['task_success'] is False else 0


if __name__=='__main__':
    raise SystemExit(main())


