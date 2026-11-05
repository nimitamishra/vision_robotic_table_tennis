import unittest,tempfile,json
import numpy as np
from intercept import Trial,predicted_crossing
from intercept_session import Session
class Tests(unittest.TestCase):
 def test_forecast_direction(self):
  r={'forecast_path':[[0,1,1],[0,0,.8]],'forecast_duration_s':.5}
  p=predicted_crossing(r,.5);self.assertAlmostEqual(p['time_s'],.25)
  self.assertIsNone(predicted_crossing({'forecast_path':list(reversed(r['forecast_path'])),'forecast_duration_s':.5},.5))
 def test_causal_metrics(self):
  tr=Trial(1,.2,1.5);record=None
  for i in range(62):
   t=i/100;p=[0,.8-t,.6];h=.8
   r={'forecast_path':[p,[0,.8-t-h,.6]],'forecast_duration_s':h}
   record=tr.observe(t,p,r)
   if record:break
  self.assertIsNotNone(record);self.assertEqual(record['status'],'observed_crossing')
  for m in record['lead_metrics']:
   self.assertEqual(m['status'],'evaluated');self.assertLess(m['position_error_m'],1e-9)
   self.assertLess(abs(m['time_error_s']),1e-9);self.assertGreaterEqual(m['actual_lead_s'],m['requested_lead_s'])
 def test_gap_cannot_be_reference(self):
  tr=Trial(1,.2,1.5);tr.observe(0,[0,.3,.5],{});self.assertIsNone(tr.observe(.3,[0,.1,.5],{}))
  self.assertEqual(tr.finish()['status'],'no_observed_crossing')
 def test_outside(self):
  tr=Trial(1,.2,1.5);tr.observe(0,[2,.25,.5],{});r=tr.observe(.05,[2,.15,.5],{})
  self.assertEqual(r['status'],'outside_test_window')
 def test_no_input_trials_count(self):
  with tempfile.TemporaryDirectory() as d:
   s=Session(d);s.start(0,.2,1.5,count=2);s.tick(3);s.tick(8);s.tick(11);s.tick(16)
   self.assertEqual(len(s.results),2);self.assertFalse(s.active);self.assertTrue(all(r['status']=='no_packets' for r in s.results))
   self.assertEqual(json.loads((s.folder/'summary.json').read_text())['attempts_completed'],2)
 def test_cancel_does_not_duplicate(self):
  with tempfile.TemporaryDirectory() as d:
   s=Session(d);s.start(0,.2,1.5);s.tick(3);s.stop();s.stop()
   self.assertEqual(len(s.results),1);self.assertEqual(s.results[0]['status'],'cancelled')
if __name__=='__main__':unittest.main()
