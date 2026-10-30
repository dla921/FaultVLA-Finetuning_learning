import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
from faultvla.hand_controller import HandController
from faultvla.stablegrasp import separated_samples,visual_labels
from faultvla.stablegrasp_policy import SeparatedPolicy

class SeparatedControlTests(unittest.TestCase):
 def test_hand_mapping_preserves_thumb_and_rejects_unrepresentable_targets(self):
  commands=np.array([[0,0],[1,.7],[.2,.2]])
  targets=HandController.targets(commands)
  np.testing.assert_allclose(HandController.encode(targets),commands)
  np.testing.assert_allclose(targets[1],[.5,-.5,.5,-.5,.5,-.5,-.3,.3])
  targets[1,2]=.4
  with self.assertRaises(ValueError):HandController.encode(targets)
  with self.assertRaises(ValueError):HandController.targets([float('nan'),0])
 def test_no_qpos_or_simulation_state_in_policy_input(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);ids=[f'episode_{i:03d}' for i in range(200)]
   (r/'manifest_200.json').write_text(json.dumps(dict(splits=dict(train=ids[:160],validation=ids[160:180],test=ids[180:]))))
   ep=r/'episodes'/ids[0];(ep/'rgb').mkdir(parents=True)
   (ep/'metadata.json').write_text(json.dumps(dict(accepted=True,images_saved=True,task_success=True,instruction='transfer')))
   xyz=np.zeros((10,3));xyz[:,0]=np.linspace(0,1,10)
   action=np.concatenate((np.zeros((10,5)),HandController.targets(np.tile([1,.7],(10,1)))),axis=1)
   # No qpos/qvel/contact arrays exist in this fixture.
   np.savez(ep/'data.npz',action=action,cup_xyz=xyz,rgb_state_index=[0,5])
   for k in [0,5]:Image.new('RGB',(224,224)).save(ep/'rgb'/f'{k:06d}.png')
   rows=list(separated_samples(r,'train',1,2))
   self.assertNotIn('state',rows[0]);self.assertNotIn('cup_xyz',rows[0])
   self.assertEqual(rows[0]['action'].shape,(5,7))
   np.testing.assert_allclose(rows[0]['action'][:,5:],np.tile([1,.7],(5,1)))
 def test_visual_labels_are_location_labels(self):
  np.testing.assert_array_equal(visual_labels([[0,0,0],[.4,0,.1],[1,0,0]]),[0,1,2])
 def test_camera_only_inference_routes_arm_and_hand(self):
  p=SeparatedPolicy.__new__(SeparatedPolicy)
  p.extract=lambda rgb,task:torch.ones(4)
  p.stats=dict(mean=[0]*7,std=[1]*7)
  p.arm_limits=np.tile([-.2,.2],(5,1))
  p.head=torch.nn.Linear(4,38)
  with torch.no_grad():p.head.weight.zero_();p.head.bias.fill_(2)
  result=p.predict(np.zeros((224,224,3),dtype=np.uint8),'transfer')
  self.assertEqual(result['arm_targets_rad'].shape,(5,5))
  np.testing.assert_allclose(result['arm_targets_rad'],.2)
  np.testing.assert_allclose(result['hand_commands'],1)
  np.testing.assert_allclose(result['hand_targets_rad'],HandController.targets(np.ones((5,2))))
  self.assertAlmostEqual(sum(result['visual_probabilities'].values()),1,places=6)
if __name__=='__main__':unittest.main()
