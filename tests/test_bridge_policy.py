import os,subprocess,sys,unittest
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
from faultvla.bridge_policy import BridgeHead
from prismatic.models.action_heads import MLPResNet
class BridgePolicyTests(unittest.TestCase):
 def test_actual_24_bridge_blocks_receive_gradients(self):
  torch.manual_seed(3)
  network=MLPResNet(num_blocks=24,input_dim=7*16,hidden_dim=16,output_dim=7,use_pro_version=True)
  head=BridgeHead(network,patches=4,reset_output=True)
  hidden=torch.randn(1,25,68,16)
  action,visual=head(hidden)
  self.assertEqual(tuple(action.shape),(1,5,7));self.assertEqual(tuple(visual.shape),(1,3))
  (action.abs().mean()+visual.square().mean()).backward()
  for parameter in [network.mlp_resnet_blocks[0].q_proj.weight,network.mlp_resnet_blocks[0].gating_factor,network.fc2.weight]:
   self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())
   self.assertGreater(float(parameter.grad.abs().sum()),0)
  self.assertTrue(all(not p.requires_grad for n,p in network.named_parameters() if 'film_gen' in n))
  with self.assertRaises(ValueError):head(torch.randn(1,24,68,16))
 def test_checkpoint_constants_do_not_depend_on_bridge_filename(self):
  env=dict(os.environ,VLA_ROBOT_PLATFORM='LIBERO')
  code="import runpy; d=runpy.run_path('VLA-Adapter/prismatic/vla/constants.py'); assert d['NUM_ACTIONS_CHUNK']==8 and d['PROPRIO_DIM']==8"
  subprocess.run([sys.executable,'-c',code,'bridge_policy.pt'],cwd=R,env=env,check=True,capture_output=True)
if __name__=='__main__':unittest.main()
