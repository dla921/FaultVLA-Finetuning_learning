"""Standalone VLA execution; no demo imports or trajectory fallback."""
import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'VLA-Adapter')]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT / 'configs/so101_vla.json')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--task')
    p.add_argument('--steps', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--resume-state', type=Path)
    p.add_argument('--reset-tracking-targets', action='store_true', help='On resume, reset position targets to measured positions to unload stored tracking error')
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Render and inspect interfaces; NO policy inference')
    p.add_argument('--allow-cross-embodiment', action='store_true', help='Acknowledge uncalibrated Panda-to-hand proxy')
    args = p.parse_args()
    if not args.viewer:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    import mujoco
    from PIL import Image
    from faultvla.bridge import Environment, ActionAdapter
    from faultvla.policy import Policy, validate_checkpoint
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.task: cfg['task'] = args.task
    if args.steps is not None: cfg['max_steps'] = args.steps
    if cfg['max_steps'] <= 0 or cfg['control_dt'] <= 0:
        raise ValueError('Steps and control_dt must be positive')
    checkpoint = args.checkpoint or ROOT / cfg['checkpoint']
    if not args.check_only:
        validate_checkpoint(checkpoint, cfg['unnorm_key'])
        if not args.allow_cross_embodiment:
            raise ValueError('LIBERO/Panda statistics do not match this robot. Read docs/INTEGRATION.md and use --allow-cross-embodiment for an experimental rollout.')
    output = args.output or ROOT / 'results' / ('vla_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    cfg['resolved_checkpoint'] = str(checkpoint.resolve())
    (output / 'config.json').write_text(json.dumps(cfg, indent=2))
    metadata = dict(expected_checkpoint_repo=cfg['checkpoint_repo'],
        expected_revision=cfg['checkpoint_revision'], checkpoint_files={})
    for name in ['config.json', 'dataset_statistics.json']:
        file = checkpoint / name
        if file.is_file():
            metadata['checkpoint_files'][name] = hashlib.sha256(file.read_bytes()).hexdigest()
    metadata['integration_source_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ['src/faultvla/bridge.py', 'src/faultvla/policy.py', 'scripts/run_vla.py']}
    metadata['mujoco_version'] = mujoco.__version__
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    robot = Environment(output / 'model')
    if not args.resume_state and 'initial_arm_rad' in cfg:
        import mujoco
        for names, values in [(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'), cfg['initial_arm_rad'])]:
            if len(names) != len(values): raise ValueError('Initial posture dimension mismatch')
            for name, value in zip(names, values):
                joint, actuator = robot.model.joint(name), robot.model.actuator(name)
                if not np.isfinite(value) or not max(joint.range[0], actuator.ctrlrange[0]) <= value <= min(joint.range[1], actuator.ctrlrange[1]):
                    raise ValueError('Initial posture outside limits: '+name)
                robot.data.joint(name).qpos[0] = value
                robot.data.ctrl[actuator.id] = value
        mujoco.mj_forward(robot.model, robot.data)
    from faultvla.snapshot import save as save_state, restore as restore_state
    resume_info = restore_state(args.resume_state, robot) if args.resume_state else None
    adapter = ActionAdapter(robot, cfg)
    if args.reset_tracking_targets:
        if not resume_info: raise ValueError('--reset-tracking-targets requires --resume-state')
        adapter.last = np.clip(robot.data.qpos[adapter.qids], adapter.low, adapter.high)
        robot.command(dict(zip(adapter.names, adapter.last)))
    if not np.isclose(cfg['control_dt'] / robot.model.opt.timestep, round(cfg['control_dt'] / robot.model.opt.timestep)):
        raise ValueError('control_dt must be a multiple of physics timestep')
    policy = None if args.check_only else Policy(checkpoint, cfg['unnorm_key'], cfg['replan_actions'])
    stats = (json.loads((ROOT / 'docs/checkpoint/dataset_statistics.json').read_text())
             if args.check_only else policy.stats)[cfg['unnorm_key']]['proprio']
    viewer = None
    if args.viewer:
        from mujoco import viewer as mv
        viewer = mv.launch_passive(robot.model, robot.data)
    queue = deque()
    summary = dict(mode='interface_check' if args.check_only else 'real_vla',
                   task_success=False, steps=0, inference_calls=0, rejected_actions=0,
                   cross_embodiment_validated=False)
    initial_z = float(robot.data.body('demo_cube').xpos[2])
    max_z = initial_z
    previous_steps = 0
    if resume_info:
        initial_z, max_z = resume_info['initial_z'], resume_info['max_z']
        previous_steps = resume_info['completed_steps']
    summary.update(resume_state=str(args.resume_state) if args.resume_state else None,
        start_simulation_time=float(robot.data.time), previous_steps=previous_steps,
        reset_tracking_targets=args.reset_tracking_targets)
    save_state(output / 'last_good.npz', robot, initial_z, max_z, previous_steps)
    try:
        with mujoco.Renderer(robot.model, height=256, width=256) as renderer, (output / 'actions.jsonl').open('w') as log:
            for step in range(1 if args.check_only else cfg['max_steps']):
                if viewer is not None and not viewer.is_running(): break
                if not queue:
                    observation = {}
                    for key, camera in [('full_image', cfg['front_camera']), ('wrist_image', cfg['wrist_camera'])]:
                        renderer.update_scene(robot.data, camera=camera)
                        observation[key] = renderer.render().copy()
                        Image.fromarray(observation[key]).save(output / f'{step:05d}_{key}.png')
                    observation['state'] = adapter.state()
                    outside = ((observation['state'] < stats['q01']) | (observation['state'] > stats['q99'])).tolist()
                    obs_report = dict(step=step, state=observation['state'].tolist(), outside_training_q01_q99=outside,
                        rgb_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in observation.items() if k != 'state'})
                    if args.check_only:
                        summary.update(obs_report)
                        print(json.dumps(summary, indent=2)); break
                    started = time.perf_counter()
                    actions = policy.predict(observation, cfg['task'])
                    queue.extend(actions)
                    summary['inference_calls'] += 1
                    log.write(json.dumps(dict(kind='inference', **obs_report, actions=actions.tolist(), seconds=time.perf_counter()-started))+'\n')
                action = queue.popleft()
                log.write(json.dumps(dict(kind='action_attempt', step=step, action=action.tolist()))+'\n')
                log.flush()
                command, report = adapter.execute(action, cfg['control_dt'], viewer)
                if not report['accepted']:
                    queue.clear()  # stale actions must not follow rejected pose
                    summary['rejected_actions'] += 1
                log.write(json.dumps(dict(kind='control', step=step, simulation_time=robot.data.time,
                    action=action.tolist(), motor_targets_rad=command.tolist(),
                    motor_positions_rad=robot.data.qpos[adapter.qids].tolist(),
                    motor_velocities_rad_s=robot.data.qvel[adapter.vids].tolist(),
                    cup_xyz=robot.data.body('demo_cube').xpos.tolist(),
                    cup_upright_cos=float(robot.data.body('demo_cube').xmat.reshape(3,3)[2,2]),
                    cup_contacts=robot.cube_contacts(), **report))+'\n')
                log.flush()
                summary['steps'] += 1
                max_z = max(max_z, float(robot.data.body('demo_cube').xpos[2]))
                save_state(output / 'last_good.npz', robot, initial_z, max_z, previous_steps+summary['steps'])
            pos = robot.data.body('demo_cube').xpos.copy()
            contacts = robot.cube_contacts()
            up = float(robot.data.body('demo_cube').xmat.reshape(3,3)[2,2])
            summary.update(final_cup_xyz=pos.tolist(), lift_m=max_z-initial_z, upright_cos=up, contacts=contacts)
            summary['task_success'] = bool(not args.check_only and max_z-initial_z > .015 and abs(pos[0]-.26)<.03
                and abs(pos[1]-.08)<.025 and up>.95 and 'table2' in contacts
                and not any(c and c.startswith('contact_') for c in contacts))
    except Exception as exc:
        summary['error'] = repr(exc)
        if hasattr(adapter, 'last_stop'):
            summary['speed_stop'] = adapter.last_stop
        raise
    finally:
        summary['end_simulation_time'] = float(robot.data.time)
        save_state(output / 'final_state.npz', robot, initial_z, max_z, previous_steps+summary['steps'])
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        if viewer: viewer.close()
    print(f'Results: {output}')
    return 0 if args.check_only or summary['task_success'] else 2

if __name__ == '__main__':
    raise SystemExit(main())



