"""Numerical contract checks, no hardware transport."""
import unittest
import torch
from common import *
from train_prior import load
class ForehandTests(unittest.TestCase):
    def test_clip_split_and_provenance(self):
        a=np.load(RUN/'paired.npz',allow_pickle=False);meta=json.loads((RUN/'dataset.json').read_text())
        sets=[set(a[f'{name}_ids']) for name in ('train','validation','test')]
        self.assertEqual([len(s) for s in sets],[16,2,2]);self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(set.union(*sets),set(range(20)));self.assertEqual(a['x'].shape,(2520,4));self.assertEqual(a['token'].shape,(2520,64))
        for row in meta['motions']:self.assertEqual(sha(GRID/row['motion']/'reference.npz'),row['npz_sha256'])
    def test_future_timing_and_stop(self):
        m=motions()[0];q,dq=future(m,.2)
        np.testing.assert_allclose(q,m['dof'][np.arange(10)*5+10],atol=1e-6)
        q,dq=future(m,3.);np.testing.assert_allclose(q,np.tile(m['dof'][-1],(10,1)));self.assertFalse(dq.any())
    def test_physical_geometry_matches_source(self):
        r=json.loads((RUN/'source_validation.json').read_text())
        self.assertLess(r['mujoco_vs_urdf_fk_max_error_m'],1e-5);self.assertLess(r['target_max_error_m'],.002)
        m=model();self.assertAlmostEqual(m.geom_pos[m.geom('table_top').id,0]-m.geom_size[m.geom('table_top').id,0],.36)
        self.assertAlmostEqual(m.geom_pos[m.geom('table_top').id,2]+m.geom_size[m.geom('table_top').id,2],.76)
    def test_prediction_domain_and_checkpoint(self):
        torch.set_num_threads(1);p=load(RUN/'policy.pt');m=motions()[0];token=p(np.r_[m['target'],1.])
        self.assertEqual(token.shape,(64,));self.assertTrue(np.isfinite(token).all())
        with self.assertRaises(ValueError):p([.8,-.38,.08,1])
        with self.assertRaises(ValueError):p([.43,-.38,.08,2])
        with self.assertRaises(ValueError):p([np.nan,-.38,.08,1])
if __name__=='__main__':unittest.main()
