import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faultvla.bridge import Environment, ActionAdapter
from faultvla.policy import validate_checkpoint

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.robot = Environment(self.tmp.name)
        self.cfg = json.loads((ROOT/'configs/so101_vla.json').read_text(encoding='utf-8-sig'))
        self.adapter = ActionAdapter(self.robot, self.cfg)

    def test_named_motor_mapping_and_state(self):
        self.assertEqual(len(self.adapter.aids), 13)
        self.assertEqual(len(set(self.adapter.aids)), 13)
        self.assertEqual(self.adapter.state().shape, (8,))
        self.assertTrue(np.isfinite(self.adapter.state()).all())

    def test_invalid_actions(self):
        for action in [np.zeros(13), np.full(7, np.nan), np.full(7, np.inf)]:
            with self.assertRaises(ValueError): self.adapter.targets(action, .05)

    def test_policy_translation_and_rotation_affect_motor_targets(self):
        # Generate a reachable action from an independent FK perturbation.
        a = self.adapter
        p0, r0 = a.pose()
        dd = mujoco.MjData(self.robot.model); dd.qpos[:] = self.robot.data.qpos
        dd.qpos[a.qids[0]] += .01
        mujoco.mj_forward(self.robot.model, dd)
        p1, r1 = a.pose(dd)
        action = np.r_[(p1-p0)/.05, Rotation.from_matrix(r1@r0.T).as_rotvec()/.5, 1.]
        start = a.last.copy()
        command, report = a.targets(action, .05)
        self.assertTrue(report['accepted'])
        self.assertGreater(np.linalg.norm(command[:5]-start[:5]), 1e-4)
        np.testing.assert_allclose(report['target_position_m'], p1)
        np.testing.assert_allclose(report['target_rotation_matrix'], r1, atol=1e-8)
        self.assertTrue(np.all(np.abs(command-start) <= a.speed*.05+1e-9))
        self.assertTrue(np.all(command >= a.low) and np.all(command <= a.high))

    def test_gripper_polarity(self):
        opened = ActionAdapter(self.robot, self.cfg)
        closed = ActionAdapter(self.robot, self.cfg)
        co, ro = opened.targets([0,0,0,0,0,0,1], .05)
        cc, rc = closed.targets([0,0,0,0,0,0,0], .05)
        self.assertTrue(ro['accepted'] and rc['accepted'])
        self.assertFalse(np.allclose(co[5:], cc[5:]))

    def test_unreachable_holds_all_motors(self):
        self.cfg['translation_scale_m'] = 10
        a = ActionAdapter(self.robot, self.cfg)
        start = a.last.copy()
        command, report = a.targets([1,1,1,0,0,0,0], .05)
        self.assertFalse(report['accepted'])
        np.testing.assert_array_equal(command, start)

    def test_frame_conversion(self):
        self.cfg['world_to_policy_rotation'] = [[0,-1,0],[1,0,0],[0,0,1]]
        a = ActionAdapter(self.robot, self.cfg)
        p, _ = a.pose()
        _, report = a.targets([.01,0,0,0,0,0,1], .05)
        np.testing.assert_allclose(report['target_position_m'], p+[0,-.0005,0])

    def test_per_physics_step_target_slew_and_tracking_guard(self):
        a = self.adapter
        original_step = self.robot.step
        previous = self.robot.data.ctrl[a.aids].copy()
        checks = []
        def checked_step(seconds, viewer=None):
            nonlocal previous
            now = self.robot.data.ctrl[a.aids].copy()
            checks.append(np.max(np.abs(now-previous)-a.speed*seconds))
            previous = now
            original_step(seconds, viewer)
        self.robot.step = checked_step
        _, report = a.execute([0,0,0,0,0,0,0], .05)
        self.assertEqual(len(checks), 50)
        self.assertLessEqual(max(checks), 1e-9)
        self.assertIn('tracking_limited', report)

    def test_rotation_reference_preserves_rotation_across_pi(self):
        self.cfg['orientation_reference_rotvec_rad'] = [np.pi, 0, 0]
        a = ActionAdapter(self.robot, self.cfg)
        p, current = a.pose()
        desired = Rotation.from_rotvec([3.2, 0, 0]).as_matrix()
        a.tool = current.T @ desired
        state = a.state()
        self.assertAlmostEqual(float(state[3]), 3.2, places=5)
        np.testing.assert_allclose(Rotation.from_rotvec(state[3:6]).as_matrix(), desired, atol=1e-6)

    def test_physical_steps(self):
        a = self.adapter
        for _ in range(5):
            command, report = a.execute([0,0,0,0,0,0,1], .05)
            self.assertTrue(np.isfinite(self.robot.data.qpos).all())
        self.assertAlmostEqual(self.robot.data.time, .25)

    def test_actual_overspeed_stops(self):
        self.cfg['actual_speed_stop_rad_s'] = 1e-12
        a = ActionAdapter(self.robot, self.cfg)
        with self.assertRaises(RuntimeError): a.execute([0,0,0,0,0,0,1], .05)

    def test_missing_checkpoint_is_error(self):
        with self.assertRaises(FileNotFoundError): validate_checkpoint(self.tmp.name, 'libero_spatial_no_noops')

if __name__ == '__main__': unittest.main(verbosity=2)


