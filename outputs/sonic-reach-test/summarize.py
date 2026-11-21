"""Summarize actual simulated tracking, never infer success from target motion."""
import json,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
rows=[json.loads(x) for x in (a.run/'samples.jsonl').read_text().splitlines()]
config=json.loads((a.run/'config.json').read_text()) if (a.run/'config.json').exists() else {}
ramp=float(config.get('ramp_seconds',2));hold_seconds=float(config.get('hold_seconds',5))
reach=[r for r in rows if r['phase']=='reach'];complete=[r for r in rows if r['phase']=='complete']
report={'simulation_only':True,'requested_displacement_m':rows[-1].get('requested_displacement_m',[.05,0,0]),'ramp_duration_s':ramp,'hold_duration_s':hold_seconds,'absolute_right_target_m':config.get('absolute_right_target_m'),'right_joint_goal_rad':config.get('right_joint_goal_rad'),'initial_right_q_rad':config.get('initial_right_q_rad'),'hold_initial_upper_body':config.get('hold_initial_upper_body',False),'hold_initial_vr':config.get('hold_initial_vr',False),'startup_joint_feedback':config.get('startup_joint_feedback',False),'sim_right_arm_direct_servo':config.get('sim_right_arm_direct_servo',False),'sim_arm_gravity_comp':config.get('sim_arm_gravity_comp',False),'sim_online_arm_ik':config.get('sim_online_arm_ik',False),'sim_right_arm_joint_feedback':config.get('sim_right_arm_joint_feedback',False),'sim_right_arm_cartesian_feedback':config.get('sim_right_arm_cartesian_feedback',False),'standing_seconds':config.get('standing_seconds',6),'stage_length_m':config.get('stage_length_m'),'stage_count':rows[-1].get('stage_count',1),'freeze_reach_feedback':config.get('freeze_reach_feedback',False),'reference':'SONIC offset hand point relative to pelvis; not racket center','completed':bool(complete),'virtual_support':any(r.get('virtual_support',False) for r in rows if r['phase'] in ['hold','reach','complete'])}
if (a.run/'online_ik.json').exists():
 report['online_ik']=json.loads((a.run/'online_ik.json').read_text())
report['sim_arm_extra_kp']=config.get('sim_arm_extra_kp',0)
report['sim_arm_extra_kd']=config.get('sim_arm_extra_kd',0)
report['control_input']='recorded_external_tokens' if config.get('token_replay') else 'planner_targets'
if (a.run/'token_replay_result.json').exists():
 report['token_replay']=json.loads((a.run/'token_replay_result.json').read_text())
 report['token_replay']['limitation']='Interface replay only; no learned ball-to-token policy and no physical robot validation.'
if not reach and config.get('absolute_right_target_m') is not None:
 report['requested_displacement_m']=None
if config.get('initial_right_q_rad') is not None:
 initial=np.asarray(config['initial_right_q_rad'],float)
 startup=[r for r in rows if r['phase'] not in ('reach','complete') and r.get('right_joint_positions_rad') is not None]
 if startup:
  report['startup_max_right_joint_gap_rad']=float(max(np.max(np.abs(np.asarray(r['right_joint_positions_rad'])-initial)) for r in startup))
  report['last_pre_reach_joint_gap_rad']=float(np.max(np.abs(np.asarray(startup[-1]['right_joint_positions_rad'])-initial)))
hold=[r for r in rows if r['phase']=='hold']
if hold:
 tail=[r for r in hold if r['time']>=hold[-1]['time']-5]
 report['baseline_hold_mean_errors_cm']=dict(zip(['left','right','head'],np.mean([np.linalg.norm(np.array(r['actual_local_m'])-r['target_local_m'],axis=1)*100 for r in tail],axis=0).tolist()))
 report['hold_ready_observed']=any(r.get('ready_for_reach',False) for r in hold)
 report['position_feedback']=hold[-1].get('position_feedback',False)
 report['hold_min_root_height_m']=min(r['root_height_m'] for r in hold)
 if hold[-1].get('right_joint_positions_rad') is not None:report['baseline_right_q_rad']=hold[-1]['right_joint_positions_rad']
if (a.run/'finished.json').exists():
 report['exit_phase']=json.loads((a.run/'finished.json').read_text())['phase']
 report['completed']=report['completed'] or report['exit_phase']=='complete'
report['stages_completed']=report['stage_count'] if report['completed'] else max(0,rows[-1].get('stage_index',0)-1)
if report.get('exit_phase')=='stage_tracking_failed':report['failed_stage']=rows[-1].get('stage_index')
if reach:
 t0=reach[0]['time'];start=np.array(reach[0]['actual_local_m'][1]);end=(reach+complete)[-1]['time'];steady=[r for r in reach+complete if r['time']>=end-1]
 report['starting_error_cm']=reach[0].get('right_error_cm')
 if steady:
  report['mean_final_error_cm']=float(np.mean([r['right_error_cm'] for r in steady]));report['max_final_error_cm']=max(r['right_error_cm'] for r in steady)
  measured=np.mean([r['actual_local_m'][1] for r in steady],axis=0)-start
  report['measured_displacement_m']=measured.tolist()
  requested=np.asarray(report['requested_displacement_m'],float)
  if np.linalg.norm(requested)>.001:
   report['movement_completion_ratio']=float(np.dot(measured,requested)/np.dot(requested,requested))
 report['min_root_height_m']=min(r['root_height_m'] for r in reach)
 report['max_root_height_m']=max(r['root_height_m'] for r in reach)
 if all('feet_in_contact' in r for r in reach):report['both_feet_contact_fraction']=float(np.mean([r['feet_in_contact']==2 for r in reach]))
 if all('right_arm_contacts' in r for r in reach):
  report['right_arm_contact_fraction']=float(np.mean([bool(r['right_arm_contacts']) for r in reach]))
  report['max_right_arm_contact_penetration_mm']=float(max([max(0.,-contact[2]*1000) for r in reach for contact in r['right_arm_contacts']] or [0.]))
  # The stock simulator has Dex3 fingers, whereas the physical robot has a
  # two-finger gripper. Keep the raw statistic, but show which contact model
  # produced it so it is not mistaken for a measured racket collision.
  pairs=sorted({tuple(contact[:2]) for r in reach for contact in r['right_arm_contacts']})
  report['right_arm_contact_pairs']=[list(pair) for pair in pairs]
  non_dex3=[[contact for contact in r['right_arm_contacts']
             if not any('hand_' in body for body in contact[:2])] for r in reach]
  report['right_arm_non_dex3_contact_fraction']=float(np.mean([bool(contacts) for contacts in non_dex3]))
  report['max_right_arm_non_dex3_contact_penetration_mm']=float(max(
   [max(0.,-contact[2]*1000) for contacts in non_dex3 for contact in contacts] or [0.]))
 if all('right_joint_motor_torque_nm' in r and 'right_joint_torque_limit_nm' in r for r in reach):
  report['max_right_arm_torque_limit_fraction']=float(max(np.max(np.abs(np.asarray(r['right_joint_motor_torque_nm']))/np.asarray(r['right_joint_torque_limit_nm'])) for r in reach))
 if all('reach_joint_correction_rad' in r for r in reach):
  report['max_reach_joint_correction_rad']=float(max(np.max(np.abs(r['reach_joint_correction_rad'])) for r in reach))
 if all('correction_m' in r for r in reach):report['right_correction_at_limit_fraction']=float(np.mean([np.linalg.norm(r['correction_m'][1])>=.0399 for r in reach]))
 report['final_window_samples']=len(steady)
 report['final_window_span_s']=steady[-1]['time']-steady[0]['time'] if steady else 0.
 report['within_2cm_for_final_second']=report['completed'] and len(steady)>=15 and report['final_window_span_s']>=.8 and all(r['right_error_cm']<=2 for r in steady)
 if .01<=np.linalg.norm(np.asarray(report['requested_displacement_m'],float))<=.05:
  report['short_reach_success']=bool(report['within_2cm_for_final_second'] and report.get('movement_completion_ratio',0)>=.7 and report['max_final_error_cm']<=1.0 and report.get('both_feet_contact_fraction',0)>=.99 and report.get('max_right_arm_contact_penetration_mm',0)<=1. and (not config.get('sim_online_arm_ik') or 'max_right_arm_contact_penetration_mm' in report))
else:report['reason']='No reach phase was recorded.'
(a.run/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
