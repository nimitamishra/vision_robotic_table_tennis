import unittest
from checks import settled
class Checks(unittest.TestCase):
 def test_ready_requires_stable_history(self):
  good=[(i*.05,1.) for i in range(21)]
  self.assertTrue(settled(good,1.))
  self.assertFalse(settled(good[-5:],1.))
  self.assertFalse(settled(good,2.))
  self.assertFalse(settled(good[:-1]+[(1.,3.)],1.))
  self.assertFalse(settled(good[:-1]+[(1.,float('nan'))],1.))
if __name__=='__main__':unittest.main()
