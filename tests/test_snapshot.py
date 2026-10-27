import sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from faultvla.bridge import Environment
from faultvla.snapshot import save,restore

class SnapshotTests(unittest.TestCase):
 def test_restore_and_next_physics_step(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Environment(Path(tmp)/'a');r.step(.05)
   save(Path(tmp)/'state.npz',r,.122,.13,1)
   other=Environment(Path(tmp)/'b');info=restore(Path(tmp)/'state.npz',other)
   self.assertEqual(info['completed_steps'],1)
   np.testing.assert_array_equal(r.data.qpos,other.data.qpos)
   np.testing.assert_array_equal(r.data.qvel,other.data.qvel)
   np.testing.assert_array_equal(r.data.ctrl,other.data.ctrl)
   self.assertEqual(r.data.time,other.data.time)
   r.step(.05);other.step(.05)
   np.testing.assert_allclose(r.data.qpos,other.data.qpos,rtol=0,atol=1e-10)
   np.testing.assert_allclose(r.data.qvel,other.data.qvel,rtol=0,atol=1e-10)

if __name__=='__main__':unittest.main()
