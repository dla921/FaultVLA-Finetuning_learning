import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'VLA-Adapter')]
from faultvla.policy import Policy

class PolicyRoutingTests(unittest.TestCase):
    def test_real_inference_wrapper_receives_images_state_and_task(self):
        # Stub only the unavailable model for an explicitly labelled unit test.
        # The production runner has no stub or fallback policy.
        p = Policy.__new__(Policy)
        p.cfg = SimpleNamespace()
        p.model = object(); p.processor = object(); p.head = object(); p.proprio = object()
        obs = dict(full_image=np.zeros((256,256,3), np.uint8),
                   wrist_image=np.ones((256,256,3), np.uint8), state=np.arange(8,dtype=float))
        def inference(cfg, model, processor, observation, task, **kw):
            self.assertIs(model, p.model)
            self.assertEqual(task, 'move the cup')
            np.testing.assert_array_equal(observation['wrist_image'], obs['wrist_image'])
            np.testing.assert_array_equal(observation['state'], obs['state'])
            observation['state'][:] = -1
            return [np.array([.1,.2,.3,0,0,0,1])]
        p.infer = inference
        actions = p.predict(obs, 'move the cup')
        np.testing.assert_array_equal(obs['state'], np.arange(8))
        np.testing.assert_allclose(actions[0], [.1,.2,.3,0,0,0,1])

    def test_upstream_normalization(self):
        from experiments.robot.openvla_utils import normalize_proprio
        stats = json.loads((ROOT/'docs/checkpoint/dataset_statistics.json').read_text())['libero_spatial_no_noops']['proprio']
        lo, hi = np.array(stats['q01']), np.array(stats['q99'])
        np.testing.assert_allclose(normalize_proprio((lo+hi)/2, stats), 0, atol=1e-5)
        np.testing.assert_allclose(normalize_proprio(lo-1, stats), -1)
        np.testing.assert_allclose(normalize_proprio(hi+1, stats), 1)

if __name__ == '__main__': unittest.main(verbosity=2)
