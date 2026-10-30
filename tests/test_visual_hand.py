import sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from faultvla.visual_hand import HandSupervisor,stage_labels
from faultvla.hand_controller import HandController
class VisualHandTests(unittest.TestCase):
 def test_stage_labels_are_offline_and_settle_uses_previous_stage(self):
  np.testing.assert_array_equal(stage_labels(['00_center_wait','settle','02_align','02_preclose_settle','settle','04_lift','07_release','settle']),[0,0,0,1,1,2,3,3])
  with self.assertRaises(ValueError):stage_labels(['unknown'])
 def test_single_frame_does_not_close_and_false_resets_counter(self):
  hand=HandController.targets([0,0]);proposal=np.zeros((5,2));s=HandSupervisor(confirmations=2)
  self.assertIsNone(s.update([0,1,0,0],proposal,hand)['event'])
  self.assertEqual(s.update([1,0,0,0],proposal,hand)['ready_count'],0)
  np.testing.assert_array_equal(s.update([1,0,0,0],proposal,hand)['targets'],HandController.targets(proposal))
 def test_camera_takeover_closes_even_when_vla_keeps_open(self):
  previous=HandController.targets([0,0]);s=HandSupervisor(confirmations=2,speed=1);proposal=np.zeros((5,2))
  first=s.update([0,1,0,0],proposal,previous);previous=first['targets'][-1]
  second=s.update([0,1,0,0],proposal,previous)
  self.assertEqual(second['event'],'close_command_triggered')
  for target in second['targets']:
   self.assertTrue(np.all(np.abs(target-previous)<=.02+1e-12));previous=target
  for _ in range(20):
   result=s.update([0,0,1,0],proposal,previous);previous=result['targets'][-1]
  self.assertEqual(s.state,'holding');np.testing.assert_allclose(previous,[.5,-.5,.5,-.5,.5,-.5,-.3,.3])
  # A release image once cannot immediately release.
  self.assertIsNone(s.update([0,0,0,1],proposal,previous)['event'])
  result=s.update([0,0,0,1],proposal,previous)
  self.assertEqual(result['event'],'release_command_triggered')
 def test_uncertain_images_never_force_close(self):
  s=HandSupervisor();proposal=np.zeros((5,2));previous=HandController.targets([0,0])
  for _ in range(30):s.update([.25]*4,proposal,previous)
  self.assertEqual(s.state,'approach');self.assertEqual(s.events,[])
if __name__=='__main__':unittest.main()
