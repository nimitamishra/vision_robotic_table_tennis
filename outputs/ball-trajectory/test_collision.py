import unittest
import numpy as np
from collision import forecast
S=dict(width_m=1.525,length_m=2.74,height_m=.76,tag_center_inset_m=.0265,net_height_m=.1525)
class Tests(unittest.TestCase):
 def test_table_bounce(self):
  r=forecast([0,.2,1],[0,.3,0],S,horizon=.8)
  e=r['collision_events'][0];self.assertEqual(e['type'],'table');self.assertAlmostEqual(e['time_s'],np.sqrt(2*.98/9.81))
  self.assertAlmostEqual(e['ball_center_m'][2],.02);self.assertEqual(e['surface_point_m'][2],0)
  self.assertAlmostEqual(e['outgoing_velocity_m_s'][2],-e['incoming_velocity_m_s'][2]*.8)
  self.assertTrue(any(s['phase']=='after_bounce' for s in r['collision_segments']))
  self.assertTrue(all(p[2]>=.02-1e-8 for s in r['collision_segments'] for p in s['points']))
 def test_misses_table(self):
  r=forecast([2,.2,.5],[0,0,0],S,horizon=.8)
  self.assertEqual(r['collision_events'][0]['type'],'floor');self.assertFalse(any(e['type']=='table' for e in r['collision_events']))
 def test_net_both_directions(self):
  for y,vy in [(1.,3.),(1.7,-3.)]:
   r=forecast([0,y,.15],[0,vy,0],S,horizon=.5)
   self.assertEqual(r['collision_events'][0]['type'],'net_risk');self.assertEqual(len(r['collision_segments']),1)
 def test_clear_net(self):
  r=forecast([0,1.,1.],[0,3.,0],S,horizon=.2)
  self.assertFalse(r['collision_events'])
 def test_edge_uncertain(self):
  r=forecast([.76,.2,.3],[0,0,0],S,horizon=.5)
  self.assertEqual(r['collision_events'][0]['type'],'table_edge_risk');self.assertEqual(len(r['collision_segments']),1)
 def test_horizon(self):
  r=forecast([0,.2,1],[0,0,0],S,horizon=.1);self.assertFalse(r['collision_events']);self.assertAlmostEqual(r['collision_forecast_duration_s'],.1)
 def test_upward_surface(self):
  r=forecast([0,.2,.02],[0,0,2],S,horizon=.5)
  self.assertAlmostEqual(r['collision_events'][0]['time_s'],4/9.81)
 def test_second_contact_stops(self):
  r=forecast([0,.2,.2],[0,0,0],S,horizon=1,restitution=.5)
  self.assertEqual(len(r['collision_events']),2);self.assertEqual(r['collision_forecast_limit'],'table')
 def test_shifted_table(self):
  r=forecast([0,.2,1.3],[0,0,0],S,horizon=.8,table_z=.3)
  self.assertAlmostEqual(r['collision_events'][0]['ball_center_m'][2],.32)
 def test_slight_penetration_descending(self):
  pos=np.array([.25135793,.39388472,.01831017]);saved=pos.copy()
  r=forecast(pos,[.36383571,-3.17479874,-2.93768602],S,restitution=.9,tangent_retention=.6)
  np.testing.assert_array_equal(pos,saved)
  e=r['collision_events'][0]
  self.assertEqual(e['type'],'table');self.assertEqual(e['time_s'],0.)
  self.assertGreater(e['outgoing_velocity_m_s'][2],0)
  self.assertAlmostEqual(r['collision_model']['contact_adjustment_m'],.00168983)
  self.assertTrue(all(p[2]>=.02-1e-8 for seg in r['collision_segments'] for p in seg['points']))
 def test_exact_surface_descending(self):
  r=forecast([0,.2,.02],[0,0,-2],S)
  self.assertEqual(r['collision_events'][0]['time_s'],0.)
  self.assertEqual(r['collision_events'][0]['type'],'table')
 def test_slight_penetration_ascending(self):
  r=forecast([0,.2,.018],[0,0,2],S)
  self.assertAlmostEqual(r['collision_events'][0]['time_s'],4/9.81)
 def test_large_penetration_is_invalid(self):
  r=forecast([0,.2,.014],[0,0,-2],S)
  self.assertEqual(r['collision_forecast_limit'],'below_table_surface')
  self.assertFalse(r['collision_events']);self.assertFalse(r['collision_segments'])
 def test_penetration_outside_table_not_snapped(self):
  r=forecast([2,.2,.018],[0,0,-2],S)
  self.assertEqual(r['collision_events'][0]['type'],'floor')
  self.assertEqual(r['collision_model']['contact_adjustment_m'],0.)
 def test_penetration_near_edge_stays_risk(self):
  r=forecast([.76,.2,.018],[0,0,-2],S)
  self.assertEqual(r['collision_events'][0]['type'],'table_edge_risk')
  self.assertEqual(len(r['collision_segments']),1)
 def test_low_energy_and_invalid(self):
  r=forecast([0,.2,.3],[0,0,0],S,horizon=.5,restitution=0);self.assertEqual(r['collision_forecast_limit'],'low_energy_contact')
  with self.assertRaises(ValueError):forecast([0,0,1],[0,0,0],S,restitution=2)
if __name__=='__main__':unittest.main()
