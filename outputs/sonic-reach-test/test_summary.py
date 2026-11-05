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
if __name__=='__main__':unittest.main()
