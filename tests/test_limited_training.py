import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from faultvla.limited_training import (CHUNK, STATS_KEY, bounds, chunk_indices,
    normalize_actions, denormalize_actions, training_statistics, examples)


class LimitedTrainingTests(unittest.TestCase):
    def test_chunk_does_not_cross_episode_and_sampling_covers_task(self):
        indices = chunk_indices(470, maximum=16)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 470-CHUNK)
        self.assertEqual(len(set(indices)), 16)
        self.assertTrue(all(i+CHUNK <= 470 for i in indices))
        self.assertEqual(chunk_indices(7), [])

    def test_gripper_stays_binary_and_action_roundtrip(self):
        values = np.tile(np.linspace(-.1,.1,101)[:,None],(1,7))
        values[:,6] = (np.arange(101)>50).astype(float)
        stats = bounds(values,[True]*6+[False])
        normalized = normalize_actions(values,stats,clip=False)
        np.testing.assert_array_equal(normalized[:,6],values[:,6])
        np.testing.assert_allclose(denormalize_actions(normalized,stats),values,atol=1e-7)
        self.assertTrue(np.isfinite(normalize_actions(np.zeros((2,7)),bounds(np.zeros((3,7)),[True]*6+[False]))).all())

    def test_statistics_read_train_only_and_reject_wrong_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for split,value in [('train',.1),('validation',1000.)]:
                ep=base/split;ep.mkdir()
                (base/(split+'.json')).write_text(json.dumps([dict(id=split,path=split,split=split)]))
                (ep/'metadata.json').write_text(json.dumps(dict(success=True,error=None,rejected=0,clipped=0)))
                np.savez(ep/'trajectory.npz',actions_7d=np.full((10,7),value),state_8d=np.full((10,8),value))
            stats,ids=training_statistics(base)
            self.assertEqual(ids,['train'])
            np.testing.assert_allclose(stats[STATS_KEY]['proprio']['mean'],.1)
            with self.assertRaises(ValueError):
                list(examples(base,'train',selected=['validation']))


if __name__ == '__main__':
    unittest.main()
