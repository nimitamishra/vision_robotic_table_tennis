"""Checks that catch attachment, joint mapping, time and collision regressions."""
import unittest
import numpy as np
import mujoco
from environment import HittingPlaneEnv
from mount import model,CONFIG
from train_residual import BASE
from feed import FeedPlanner

class PipelineTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.env=HittingPlaneEnv(BASE)
 def test_attachment_removes_mass_and_dofs(self):
  m=self.env.m
  self.assertEqual(m.nu,29);self.assertEqual(m.nq,43)
  self.assertFalse(any('_hand_' in m.body(i).name for i in range(m.nbody)))
  self.assertAlmostEqual(float(m.body('right_wrist_yaw_link').mass[0]),.08457647)
  self.assertAlmostEqual(float(m.body('training_racket').mass[0]),.2)
 def test_body_seed_mapping_by_name(self):
  e=self.env.base;e.reset(np.zeros(3))
  for j in range(e.model.njnt):
   name=e.model.joint(j).name;old=mujoco.mj_name2id(e.seed_model,mujoco.mjtObj.mjOBJ_JOINT,name)
   if old<0 or e.model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE:continue
   self.assertEqual(e.data.qpos[e.model.jnt_qposadr[j]],e.seed['qpos'][e.seed_model.jnt_qposadr[old]])
 def test_input_contract(self):
  p=self.env.policy
  self.assertEqual(p.predict([.43,-.38,.178],1).shape,(64,))
  for xyz,t in [([.7,-.38,.178],1),([.43,-.38,.178],2),([.43,np.nan,.178],1)]:
   with self.assertRaises(ValueError):p.predict(xyz,t)
 def test_kinematics_matches_public_references(self):
  import common as fh
  e=self.env.base;errors=[]
  for clip in fh.motions():
   for k in (0,25,50,75,100,125):
    e.data.qpos[:3]=clip['root_pos'][k];e.data.qpos[3:7]=clip['root_wxyz'][k];e.data.qpos[e.bqa]=clip['dof'][k]
    mujoco.mj_forward(e.model,e.data)
    errors.append(np.linalg.norm(fh.face(e)-clip['actual_contact_pos'][k]))
  self.assertLess(max(errors),1e-5)
 def test_real_ball_contact_debounce_and_launch(self):
  e=self.env;e.run()
  self.assertEqual(e.own_bounces,1)
  self.assertLess(e.actual_launch_time-e.launch_time,.00101)
  # A table contact persists for multiple physics steps; it is not an obstacle.
  self.assertTrue(any(ev['type']=='table' for ev in e.events))
  e.reset();e.launched=True;e.previous_ball=e.ball();e.failure=None
  # Use a real ball/table impact and inspect the entire repeated-contact interval.
  e.d.qpos[e.qa:e.qa+3]=[1.,-.38,.782];e.d.qvel[e.va:e.va+3]=[0,0,-1]
  mujoco.mj_forward(e.m,e.d)
  e.d.time=.5
  for _ in range(15):mujoco.mj_step(e.m,e.d);e.contacts()
  self.assertIsNone(e.failure)
if __name__=='__main__':unittest.main()
