import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from faultvla.command_limiter import CommandLimiter
class LimiterTests(unittest.TestCase):
 def test_rate_bound_reversal_and_no_mutation(self):
  start=np.zeros(13);limits=np.tile([-2,2],(13,1))
  limiter=CommandLimiter(start,limits,arm_speed=1,hand_speed=2)
  target=np.ones(13)
  for _ in range(7):
   previous=limiter.previous.copy();result=limiter.apply(target)
   self.assertTrue(np.all(np.abs(result-previous)<=np.r_[np.full(5,.02),np.full(8,.04)]+1e-12))
  np.testing.assert_array_equal(target,np.ones(13));np.testing.assert_array_equal(start,np.zeros(13))
  previous=limiter.previous.copy();result=limiter.apply(-target)
  np.testing.assert_allclose(previous-result,np.r_[np.full(5,.02),np.full(8,.04)])
 def test_disabled_is_original_joint_clipping(self):
  limits=np.tile([-1,1],(13,1));target=np.linspace(-3,3,13)
  limiter=CommandLimiter(np.zeros(13),limits)
  np.testing.assert_array_equal(limiter.apply(target),np.clip(target,-1,1))
  other=np.linspace(.13,-.27,13)
  np.testing.assert_array_equal(limiter.apply(other),other)
 def test_independent_groups_and_reachable_endpoint(self):
  limiter=CommandLimiter(np.zeros(13),np.tile([-1,1],(13,1)),arm_speed=1)
  result=limiter.apply(np.full(13,2.))
  np.testing.assert_allclose(result[:5],.02);np.testing.assert_array_equal(result[5:],np.ones(8))
  for _ in range(60):result=limiter.apply(np.full(13,2.))
  np.testing.assert_allclose(result,1)
 def test_reject_invalid_rate_and_target(self):
  for rate in [0,-1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):CommandLimiter(np.zeros(13),np.tile([-1,1],(13,1)),arm_speed=rate)
  limiter=CommandLimiter(np.zeros(13),np.tile([-1,1],(13,1)))
  with self.assertRaises(ValueError):limiter.apply(np.full(13,float('nan')))
  with self.assertRaises(ValueError):limiter.apply(np.zeros(5))
if __name__=='__main__':unittest.main()
