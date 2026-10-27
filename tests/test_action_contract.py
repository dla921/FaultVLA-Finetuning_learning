import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from faultvla.bridge import Environment,ActionAdapter
from faultvla.expert_data import encode_target,configure_layout
class ContractTests(unittest.TestCase):
 def test_encode_inverse_frame_and_orientation(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Environment(tmp);cfg=json.loads((ROOT/'configs/so101_vla.json').read_text(encoding='utf-8-sig'))
   cfg['world_to_policy_rotation']=[[0,-1,0],[1,0,0],[0,0,1]]
   a=ActionAdapter(r,cfg);p,R=a.pose();target=p+[.002,-.001,.003];desired=Rotation.from_rotvec([.01,.02,-.03]).as_matrix()@R
   action=encode_target(a,target,desired,1)
   np.testing.assert_allclose(p+a.frame.T@(action[:3]*.05),target,atol=1e-12)
   np.testing.assert_allclose(Rotation.from_rotvec(a.frame.T@(action[3:6]*.5)).as_matrix()@R,desired,atol=1e-12)
   self.assertEqual(action.shape,(7,));self.assertEqual(action[-1],1)
 def test_layout_reset_changes_only_object_and_destination(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Environment(tmp);q=r.data.qpos[:5].copy()
   layout=dict(cup_xyz=[.264,-.077,.122],table2_xy=[.304,.084])
   configure_layout(r,layout)
   np.testing.assert_allclose(r.data.body('demo_cube').xpos,layout['cup_xyz'])
   np.testing.assert_array_equal(r.data.qpos[:5],q)
   np.testing.assert_allclose(r.model.geom('table2').pos[:2],layout['table2_xy'])
if __name__=='__main__':unittest.main()
