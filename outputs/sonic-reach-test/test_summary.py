import json, subprocess, sys, tempfile, unittest
from pathlib import Path

class SummaryTests(unittest.TestCase):
    def test_direction_and_original_target_error_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            rows=[]
            for i in range(141):
                t=i*.05
                rows.append(dict(phase='reach' if t<7 else 'complete',time=100+t,
                    trial_elapsed_s=t,requested_displacement_m=[0,0,-.05],
                    actual_local_m=[[0,0,0],[0,0,-.04*min(t/2,1)],[0,0,0]],
                    right_error_cm=1.,root_height_m=.78,virtual_support=False))
            (run/'samples.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),str(run)],check=True,capture_output=True)
            report=json.loads((run/'report.json').read_text())
            self.assertEqual(report['requested_displacement_m'],[0,0,-.05])
            self.assertTrue(report['within_2cm_for_final_second'])
            self.assertAlmostEqual(report['measured_displacement_m'][2],-.04)
            self.assertEqual(report['mean_final_error_cm'],1.)
            # A single accurate end sample must never qualify as a full-window pass.
            (run/'samples.jsonl').write_text(json.dumps(rows[-1] | {'phase':'reach'})+'\n'+json.dumps(rows[-1]))
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),str(run)],check=True,capture_output=True)
            report=json.loads((run/'report.json').read_text())
            self.assertFalse(report['within_2cm_for_final_second'])

    def test_custom_reach_duration_uses_actual_final_second(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            (run/'config.json').write_text(json.dumps({'ramp_seconds':8,'hold_seconds':5,
                'absolute_right_target_m':[.4,0,.2]}))
            rows=[]
            for i in range(261):
                t=i*.05
                rows.append(dict(phase='reach' if t<13 else 'complete',time=100+t,
                    trial_elapsed_s=t,requested_displacement_m=[.3,0,.4],
                    actual_local_m=[[0,0,0],[.4,0,.2] if t>=12 else [0,0,0],[0,0,0]],
                    right_error_cm=0. if t>=12 else 50.,root_height_m=.78,virtual_support=False))
            (run/'samples.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),str(run)],check=True,capture_output=True)
            report=json.loads((run/'report.json').read_text())
            self.assertEqual(report['absolute_right_target_m'],[.4,0,.2])
            self.assertTrue(report['within_2cm_for_final_second'])
            self.assertEqual(report['mean_final_error_cm'],0.)

    def test_aborted_live_start_reports_joint_drift_without_fake_reach(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            (run/'config.json').write_text(json.dumps({'absolute_right_target_m':[.5,0,.2],
                'initial_right_q_rad':[0]*7}))
            rows=[dict(phase='standing',time=1,requested_displacement_m=[.05,0,0],
                right_joint_positions_rad=[0]*7),
                dict(phase='initial_posture_not_held',time=2,requested_displacement_m=[.05,0,0],
                right_joint_positions_rad=[.48,0,0,0,0,0,0])]
            (run/'samples.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),str(run)],check=True,capture_output=True)
            report=json.loads((run/'report.json').read_text())
            self.assertIsNone(report['requested_displacement_m'])
            self.assertAlmostEqual(report['startup_max_right_joint_gap_rad'],.48)

    def test_short_reach_cannot_pass_by_staying_near_the_start(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            rows=[]
            for i in range(121):
                t=i*.05
                rows.append(dict(phase='reach' if t<6 else 'complete',time=100+t,
                    requested_displacement_m=[.02,0,0],
                    actual_local_m=[[0,0,0],[.002,0,0],[0,0,0]],
                    right_error_cm=1.8,root_height_m=.78,virtual_support=False,
                    feet_in_contact=2))
            (run/'samples.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),str(run)],check=True,capture_output=True)
            report=json.loads((run/'report.json').read_text())
            self.assertTrue(report['within_2cm_for_final_second'])
            self.assertFalse(report['short_reach_success'])
            self.assertLess(report['movement_completion_ratio'],.7)

    def test_short_reach_with_arm_collision_is_not_successful(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            rows=[]
            for i in range(121):
                t=i*.05
                rows.append(dict(phase='reach' if t<6 else 'complete',time=100+t,
                    requested_displacement_m=[.02,0,0],
                    actual_local_m=[[0,0,0],[.02*min(t/2,1),0,0],[0,0,0]],
                    right_error_cm=0.,root_height_m=.78,virtual_support=False,
                    feet_in_contact=2,
                    right_arm_contacts=[['right_wrist_yaw_link','right_hip_pitch_link',-.002]]))
            (run/'samples.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),str(run)],check=True,capture_output=True)
            report=json.loads((run/'report.json').read_text())
            self.assertTrue(report['within_2cm_for_final_second'])
            self.assertEqual(report['max_right_arm_contact_penetration_mm'],2.)
            self.assertFalse(report['short_reach_success'])
if __name__=='__main__':unittest.main()
