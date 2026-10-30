import sys,unittest
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
from faultvla.command_context import CommandContext,contexts_from_actions
from faultvla.hand_controller import HandController
from faultvla.coordinated_policy import CoordinatedHead
from prismatic.models.action_heads import MLPResNet
STATS=dict(mean=[0]*7,std=[1]*7)
def command(main,thumb):
 return np.r_[np.zeros(5),[main,-main,main,-main,main,-main,thumb,-thumb]]
class CommandContextTests(unittest.TestCase):
 def test_offline_context_matches_causal_online_commands(self):
  actions=np.array([command(-.7,-.7),command(-.4,-.5),command(.5,-.3),command(-.8,-.8)])
  offline=contexts_from_actions(actions,STATS);tracker=CommandContext(STATS)
  self.assertEqual(offline[0,-1],0);np.testing.assert_array_equal(offline[0,:7],0)
  for index,action in enumerate(actions):
   np.testing.assert_array_equal(offline[index],tracker.features());tracker.observe(action)
  self.assertEqual(tracker.stage,3)
  other=actions.copy();other[2:]=command(-1,-1)
  np.testing.assert_array_equal(offline[:3],contexts_from_actions(other,STATS)[:3])
  self.assertFalse(np.array_equal(offline[3],contexts_from_actions(other,STATS)[3]))
 def test_command_stage_does_not_mean_measured_grasp(self):
  tracker=CommandContext(STATS)
  for index,action in enumerate([command(-.7,-.7),command(-.4,-.5),command(.5,-.3),command(-.8,-.8)]):
   tracker.observe(action);self.assertEqual(tracker.stage,index)
  restored=CommandContext(STATS);restored.set_features(tracker.features())
  np.testing.assert_allclose(restored.features(),tracker.features())
 def test_invalid_context_is_rejected(self):
  tracker=CommandContext(STATS)
  with self.assertRaises(ValueError):tracker.observe(np.full(13,np.nan))
  with self.assertRaises(ValueError):tracker.set_features(np.ones(12))
  with self.assertRaises(ValueError):CommandContext(dict(mean=[0]*7,std=[0]*7))
 def test_context_projection_changes_output_and_receives_gradient(self):
  torch.manual_seed(9)
  network=MLPResNet(num_blocks=24,input_dim=7*16,hidden_dim=16,output_dim=7,use_pro_version=True)
  # Nonzero gates exercise the actual conditioning path without any VLM sensor token.
  for block in network.mlp_resnet_blocks:block.gating_factor.data.fill_(.2)
  head=CoordinatedHead(network,patches=4,reset_output=True)
  hidden=torch.randn(1,25,68,16);context=torch.zeros(1,12);context[:,7]=1
  action,_=head(hidden,context);changed=context.clone();changed[:,7]=0;changed[:,9]=1
  alternative,_=head(hidden,changed)
  self.assertGreater(float((action-alternative).abs().max()),0)
  action.square().mean().backward()
  self.assertGreater(float(head.context_projection[0].weight.grad.abs().sum()),0)
  with self.assertRaises(ValueError):head(hidden)
if __name__=='__main__':unittest.main()
