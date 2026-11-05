import unittest
import numpy as np
from feedback import update_correction
class FeedbackTests(unittest.TestCase):
 def test_large_error_is_rate_and_amplitude_limited(self):
  c=np.zeros((3,3));e=np.ones((3,3))*10
  first=update_correction(c,e,.02)
  self.assertLessEqual(np.max(np.abs(first)),.0002+1e-10)
  for _ in range(10000):c=update_correction(c,e,.02)
  self.assertTrue(np.all(np.linalg.norm(c,axis=1)<=.04+1e-10))
 def test_stall_does_not_cause_large_jump(self):
  c=update_correction(np.zeros((3,3)),np.ones((3,3)),10)
  self.assertLessEqual(np.max(np.abs(c)),.001)
 def test_bad_measurement_rejected(self):
  with self.assertRaises(ValueError):update_correction(np.zeros((3,3)),np.full((3,3),np.nan),.02)
  with self.assertRaises(ValueError):update_correction(np.zeros((3,3)),np.zeros((3,3)),-1)
if __name__=='__main__':unittest.main()
