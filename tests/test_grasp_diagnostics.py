import json,sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from faultvla.grasp_diagnostics import observe,DiagnosticSummary
sys.path.insert(0,str(R/'scripts'))
from compare_grasp_diagnostics import compare
class GraspDiagnosticTests(unittest.TestCase):
 def test_comparison_checks_episode_and_preserves_geometry(self):
  record=dict(palm_cup_center_distance_m=.1,cup_minus_palm_hand_frame_m=[.1,0,0],actual_joint_positions_rad=[0]*13,cup_hand_contacts=[],target_tracking_error_rad=[0]*13,palm_site_world_m=[0,0,0])
  row=dict(step=0,time=.02,command=[0]*13,diagnostics=record)
  policy=dict(episode='a',mode='policy',task_success=False)
  replay=dict(episode='a',mode='replay',task_success=True)
  result=compare(policy,[row],replay,[row])
  self.assertEqual(result['policy']['closest_distance_mm'],100)
  self.assertEqual(result['same_time_mean_palm_difference_mm'],0)
  with self.assertRaises(ValueError):compare(policy,[row],dict(replay,episode='b'),[row])
 @unittest.skipUnless((R/'data/SO101_StableGrasp200/scene.xml').exists(),'Restore source data first')
 def test_read_only_geometry_and_aggregation(self):
  import mujoco
  m=mujoco.MjModel.from_xml_path(str(R/'data/SO101_StableGrasp200/scene.xml'));d=mujoco.MjData(m)
  mujoco.mj_forward(m,d);before=(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time))
  record=observe(m,d);json.dumps(record,allow_nan=False)
  np.testing.assert_array_equal(d.qpos,before[0]);np.testing.assert_array_equal(d.qvel,before[1]);np.testing.assert_array_equal(d.ctrl,before[2]);self.assertEqual(d.time,before[3])
  delta=np.array(record['cup_minus_palm_world_m'])
  self.assertAlmostEqual(np.linalg.norm(delta),record['palm_cup_center_distance_m'])
  self.assertEqual(len(record['actual_joint_positions_rad']),13)
  agg=DiagnosticSummary();agg.update(.02,record);agg.update(.04,record)
  self.assertEqual(agg.result()['closest_approach']['time'],.02)
  self.assertEqual(agg.result()['final']['time'],.04)
if __name__=='__main__':unittest.main()
