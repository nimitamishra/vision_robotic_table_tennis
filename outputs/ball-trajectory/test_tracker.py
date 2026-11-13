import unittest
import numpy as np
from tracker import Tracker,G,measurement,stereo_range_sigma

class Tests(unittest.TestCase):
    def test_ballistic_prediction(self):
        tracker=Tracker();rng=np.random.default_rng(11);errors=[];vel_errors=[]
        p0=np.array([0.,0.,3.]);v0=np.array([.8,1.6,1.])
        for i in range(55):
            t=i/60;p=p0+v0*t+.5*G*t*t
            r=tracker.step(t,p+rng.normal(0,.003,3),'contour')
            if i>15:
                vel_errors.append(np.linalg.norm(np.array(r['velocity_m_s'])-(v0+G*t)))
                for pred in r['predictions']:
                    future=t+pred['horizon_s'];truth=p0+v0*future+.5*G*future**2
                    errors.append(np.linalg.norm(np.array(pred['position_m'])-truth))
        self.assertLess(np.mean(errors),.03);self.assertLess(np.mean(vel_errors),.15)
        print('Synthetic ballistic mean prediction error (m):',np.mean(errors))
    def flying(self):
        tr=Tracker()
        for i in range(12):
            t=i/60;tr.step(t,np.array([t,t,2.])+.5*G*t*t,'contour')
        return tr
    def test_outlier_and_missing(self):
        tr=self.flying();r=tr.step(.2,[20,20,20],'box')
        self.assertFalse(r['measurement_accepted']);self.assertEqual(r['status'],'coasting')
        r=tr.step(.23,None);self.assertEqual(r['status'],'coasting');self.assertTrue(r['predictions'])
        r=tr.step(.5,None);self.assertNotIn('position_m',r);self.assertFalse(r['predictions'])
    def test_time_order(self):
        tr=self.flying();x=tr.x.copy();r=tr.step(.1,[0,0,0])
        self.assertEqual(r['status'],'out_of_order');np.testing.assert_equal(tr.x,x)
    def test_static(self):
        tr=Tracker()
        for i in range(20):r=tr.step(i/60,[0,0,.275],'contour')
        self.assertEqual(r['status'],'stationary');self.assertEqual(r['speed_m_s'],0);self.assertFalse(r['predictions'])
    def test_stop_at_contact(self):
        tr=Tracker()
        for i in range(10):
            t=i/60;r=tr.step(t,np.array([0.,t,.25])+.5*G*t*t,'contour')
        hit=r['contact_plane_crossing']['time_s']
        self.assertLess(hit,.1);self.assertTrue(all(p['horizon_s']<hit for p in r['predictions']))
    def test_bounce_reinitializes(self):
        tr=Tracker()
        for i in range(9):
            t=i/60;tr.step(t,[0,t,.18-.7*t-4.905*t*t],'contour')
        r=tr.step(.15,[0,.15,.05],'contour')
        self.assertEqual(r['event'],'possible_bounce');self.assertFalse(r['predictions'])
    def test_bounce_fit_uses_only_post_impact_points(self):
        tr=Tracker()
        for i in range(9):
            t=i/60;tr.step(t,[0,t,.18-.7*t-4.905*t*t],'contour')
        # A new, upward ballistic arc: no old downward samples in this fit.
        for dt in [0.,.03,.06]:
            r=tr.step(.15+dt,[0,.15+dt,.05+2*dt-4.905*dt*dt],'contour')
        self.assertEqual(r['status'],'tracking')
        np.testing.assert_allclose(r['velocity_m_s'],[0,1,2-9.81*.06],atol=1e-7)
        self.assertEqual(r['initialization_samples'] if 'initialization_samples' in r else 0,0)

    def test_selection(self):
        p,s=measurement({'world_measurement':True,'ball':{},'xyz_world':[1,2,3],'ball_refined':{},'xyz_world_refined':None})
        self.assertEqual(s,'box');np.testing.assert_equal(p,[1,2,3])
        p,s=measurement({'xyz_world':[1,2,3]});self.assertIsNone(p)
    def test_far_stereo_measurement_gets_more_range_uncertainty(self):
        far={'ball':{'xyz_left':[0,0,3.],
                     'left_center_px':[224,70],'right_center_px':[220,70]}}
        near={'ball':{'xyz_left':[0,0,.85],
                      'left_center_px':[286,159],'right_center_px':[271,159]}}
        self.assertGreater(stereo_range_sigma(far,'box'),.2)
        self.assertLess(stereo_range_sigma(near,'box'),.06)
        self.assertGreater(stereo_range_sigma(far,'box'),stereo_range_sigma(near,'box'))
    def test_short_dropout_coasts_through_table_bounce(self):
        tr=Tracker(table_bounds=(-.7625,.7625,-.0265,2.7135))
        tr.x=np.array([0.,.2,.08,0.,-1.,-1.])
        tr.P=np.eye(6)*.001;tr.t=0.;tr.last_seen=0.;tr.mode='tracking'
        r=tr.step(.1,None)
        self.assertEqual(r['status'],'coasting')
        self.assertEqual(r['event'],'predicted_bounce')
        self.assertFalse(r['measurement_accepted'])
        self.assertGreater(r['position_m'][2],tr.contact_z)
        self.assertGreater(r['velocity_m_s'][2],0)
    def test_initialization_gap(self):
        tr=Tracker()
        for i in range(3):tr.step(i/60,[0,0,1])
        for i in range(4,17):tr.step(i/60,None)
        r=tr.step(.29,[0,0,1]);self.assertNotIn('velocity_m_s',r)

if __name__=='__main__':unittest.main()
