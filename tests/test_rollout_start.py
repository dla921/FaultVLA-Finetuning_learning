import sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from faultvla.rollout_start import select_start,replay_prefix
class HandoffTests(unittest.TestCase):
 def test_start_selection_and_invalid_requests(self):
  phases=['approach','align','ready','ready','carry']
  self.assertEqual(select_start(phases),0)
  self.assertEqual(select_start(phases,'ready'),2)
  self.assertEqual(select_start(phases,start_index=4),4)
  for kwargs in [dict(start_phase='missing'),dict(start_index=-1),dict(start_index=5),dict(start_phase='ready',start_index=2)]:
   with self.assertRaises(ValueError):select_start(phases,**kwargs)
 @unittest.skipUnless((R/'data/SO101_StableGrasp200/scene.xml').exists(),'Restore source data')
 def test_prefix_preserves_full_dynamic_state_across_handoff(self):
  import mujoco
  root=R/'data/SO101_StableGrasp200';m=mujoco.MjModel.from_xml_path(str(root/'scene.xml'))
  a=mujoco.MjData(m);b=mujoco.MjData(m)
  with np.load(root/'episodes/episode_081/data.npz') as z:
   actions=z['action'][:10].copy()
   for d in [a,b]:
    d.qpos[:]=z['full_qpos'][0];d.qvel[m.jnt_dofadr[m.actuator_trnid[:,0]]]=z['qvel'][0]
    d.ctrl[:]=d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]];mujoco.mj_forward(m,d)
  replay_prefix(m,a,actions)
  replay_prefix(m,b,actions[:5]);replay_prefix(m,b,actions[5:])
  np.testing.assert_array_equal(a.qpos,b.qpos);np.testing.assert_array_equal(a.qvel,b.qvel)
  np.testing.assert_array_equal(a.ctrl,b.ctrl);self.assertEqual(a.time,b.time)
  with self.assertRaises(ValueError):replay_prefix(m,b,np.full((1,13),np.nan))
if __name__=='__main__':unittest.main()
