from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / 'environments/so101_wrist_side'
sys.path.insert(0, str(DEMO))
from control import Robot, ARM, HAND

class Environment(Robot):
    """Generate camera-equipped XML copies, preserving the supplied demo."""
    def __init__(self, output):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        robot = ET.parse(DEMO / 'robot.xml')
        robot.getroot().find('compiler').set('meshdir', str(DEMO / 'assets'))
        wrist = robot.find(".//body[@name='r_wrist_interface']")
        reference = Robot()
        rotation = reference.data.body('r_wrist_interface').xmat.reshape(3, 3)
        target = np.array([.036, .015, .11])
        eye = target + rotation.T @ np.array([.12, -.12, .20])
        z = (eye - target) / np.linalg.norm(eye - target)
        x = np.cross(rotation.T @ [0, 0, 1], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        ET.SubElement(wrist, 'camera', name='vla_wrist', pos=' '.join(map(str, eye)),
                      xyaxes=' '.join(map(str, np.r_[x, y])), fovy='75')
        robot.write(output / 'robot.xml')
        scene = ET.parse(DEMO / 'scene.xml')
        scene.find('include').set('file', str(output / 'robot.xml'))
        scene.write(output / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(str(output / 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        with np.load(DEMO / 'home.npz') as home:
            self.data.qpos[:] = home['qpos']
            self.data.ctrl[:] = home['ctrl']
        self.faults = {}
        mujoco.mj_forward(self.model, self.data)

class ActionAdapter:
    """No cup coordinates, task trajectory or demo phase enters this adapter."""
    def __init__(self, robot, config):
        self.robot, self.cfg = robot, config
        self.m, self.d = robot.model, robot.data
        self.names = ARM + HAND
        self.aids = np.array([self.m.actuator(n).id for n in self.names])
        joints = np.array([self.m.joint(n).id for n in self.names])
        self.qids = self.m.jnt_qposadr[joints]
        self.vids = self.m.jnt_dofadr[joints]
        self.low = np.maximum(self.m.jnt_range[joints, 0], self.m.actuator_ctrlrange[self.aids, 0]) + 1e-5
        self.high = np.minimum(self.m.jnt_range[joints, 1], self.m.actuator_ctrlrange[self.aids, 1]) - 1e-5
        self.speed = np.array(config['joint_target_speed_rad_s'])
        if self.speed.shape != (13,) or np.any(self.speed <= 0):
            raise ValueError('Expected 13 positive target slew limits')
        self.frame = np.array(config['world_to_policy_rotation'], dtype=float)
        if not np.allclose(self.frame.T @ self.frame, np.eye(3)) or not np.isclose(np.linalg.det(self.frame), 1):
            raise ValueError('Frame transform must be a proper rotation')
        self.offset = np.array(config['world_to_policy_translation_m'])
        self.tool = Rotation.from_rotvec(config['palm_to_policy_tool_rotvec_rad']).as_matrix()
        self.last = np.clip(self.d.ctrl[self.aids].copy(), self.low, self.high)

    def pose(self, data=None):
        data = self.d if data is None else data
        return (data.site('palm_target').xpos.copy(),
                data.body('r_wrist_interface').xmat.reshape(3, 3).copy() @ self.tool)

    def state(self):
        p, rotation = self.pose()
        q = self.d.qpos[self.qids[5:]]
        opened, closed = np.array(self.cfg['hand_open_rad']), np.array(self.cfg['hand_closed_rad'])
        fraction = float(np.clip(np.mean((q - opened) / (closed - opened)), 0, 1))
        # Explicit proxy, NOT actual Panda gripper measurements.
        aperture = .04 * (1 - fraction)
        rotvec = Rotation.from_matrix(self.frame @ rotation).as_rotvec()
        reference = self.cfg.get('orientation_reference_rotvec_rad')
        # Equivalent rotation-vector branches avoid a discontinuity at pi.
        # The reference is explicit; this does not change the physical rotation.
        if reference is not None and np.linalg.norm(rotvec) > 1e-8:
            axis = rotvec / np.linalg.norm(rotvec)
            candidates = np.array([rotvec + k*2*np.pi*axis for k in (-1, 0, 1)])
            rotvec = candidates[np.argmin(np.linalg.norm(candidates-np.asarray(reference), axis=1))]
        return np.r_[self.frame @ p + self.offset, rotvec,
                     aperture, -aperture].astype(np.float32)

    def targets(self, action, dt):
        a = np.asarray(action, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('Expected seven finite unnormalized LIBERO action values')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive')
        clipped = np.clip(a[:6], -1, 1)
        p, rotation = self.pose()
        target = p + self.frame.T @ (clipped[:3] * self.cfg['translation_scale_m'])
        dr = self.frame.T @ (clipped[3:6] * self.cfg['rotation_scale_rad'])
        target_rotation = Rotation.from_rotvec(dr).as_matrix() @ rotation
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = self.d.qpos
        def residual(q):
            scratch.qpos[self.qids[:5]] = q
            mujoco.mj_forward(self.m, scratch)
            pp, rr = self.pose(scratch)
            return np.r_[(pp - target) * 10,
                         Rotation.from_matrix(target_rotation @ rr.T).as_rotvec() * 2]
        # Reuse demo's bounded least-squares approach, with VLA's full orientation.
        seed = np.clip(self.d.qpos[self.qids[:5]], self.low[:5], self.high[:5])
        fit = least_squares(residual, seed, bounds=(self.low[:5], self.high[:5]), max_nfev=150)
        residual(fit.x)
        pp, rr = self.pose(scratch)
        pe = float(np.linalg.norm(pp - target))
        re = float(Rotation.from_matrix(target_rotation @ rr.T).magnitude())
        accepted = bool(fit.success and pe <= self.cfg['ik_position_tolerance_m']
                        and re <= self.cfg['ik_orientation_tolerance_rad'])
        desired = self.last.copy()
        # Reject the entire action, including release, on unreachable targets.
        if accepted:
            desired[:5] = fit.x
            desired[5:] = self.cfg['hand_open_rad'] if a[6] > .5 else self.cfg['hand_closed_rad']
        bounded = np.clip(desired, self.low, self.high)
        command = self.last + np.clip(bounded - self.last, -self.speed * dt, self.speed * dt)
        self.last = command.copy()
        return command, dict(accepted=accepted, position_error_m=pe, orientation_error_rad=re,
                             target_position_m=target.tolist(), target_rotation_matrix=target_rotation.tolist(),
                             action_clipped=bool(np.any(clipped != a[:6])),
                             target_rate_limited=bool(np.any(np.abs(command - bounded) > 1e-9)))

    def execute(self, action, dt, viewer=None):
        command, report = self.targets(action, dt)
        tracking = np.asarray(self.cfg.get('max_tracking_error_rad', [.3]*5 + [.15]*8))
        if tracking.shape != (13,) or not np.isfinite(tracking).all() or np.any(tracking <= 0):
            raise ValueError('Expected 13 positive tracking error limits')
        report['requested_motor_targets_rad'] = command.tolist()
        report['tracking_limited'] = False
        h = self.m.opt.timestep
        for _ in range(round(dt / h)):
            measured = self.d.qpos[self.qids]
            target = np.clip(command, measured-tracking, measured+tracking)
            report['tracking_limited'] |= bool(np.any(np.abs(target-command) > 1e-9))
            previous = self.d.ctrl[self.aids].copy()
            applied = np.clip(previous + np.clip(target-previous, -self.speed*h, self.speed*h), self.low, self.high)
            self.robot.command(dict(zip(self.names, applied)))
            self.robot.step(h, viewer)
            velocity = self.d.qvel[self.vids]
            if np.any(np.abs(velocity) > self.cfg['actual_speed_stop_rad_s']):
                i = int(np.argmax(np.abs(velocity)))
                self.last_stop = dict(joint=self.names[i], velocity_rad_s=float(velocity[i]),
                    limit_rad_s=float(self.cfg['actual_speed_stop_rad_s']), simulation_time=float(self.d.time),
                    position_rad=float(self.d.qpos[self.qids[i]]), target_rad=float(applied[i]))
                self.last = np.clip(self.d.qpos[self.qids], self.low, self.high)
                self.robot.command(dict(zip(self.names, self.last)))
                raise RuntimeError(f"Actual joint speed exceeded stop threshold: {self.names[i]} "
                    f"{velocity[i]:.4f} rad/s (limit {self.cfg['actual_speed_stop_rad_s']}) "
                    f"at t={self.d.time:.4f}s; simulation halted")
        self.last = self.d.ctrl[self.aids].copy()
        return self.last.copy(), report

