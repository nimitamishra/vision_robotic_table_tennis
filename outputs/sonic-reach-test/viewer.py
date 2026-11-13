"""Read-only simulation visualization, plus commands to the simulation harness."""
import argparse,json,time
from pathlib import Path
import numpy as np
import viser
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--port',type=int,default=8090);a=ap.parse_args()
config=json.loads((a.run/'config.json').read_text()) if (a.run/'config.json').exists() else {}
absolute_target=config.get('absolute_right_target_m')
ramp_seconds=float(config.get('ramp_seconds',2))
hold_seconds=float(config.get('hold_seconds',5))
movement='pelvis-relative ready target' if absolute_target is not None else '5 cm hand displacement'
server=viser.ViserServer(host='127.0.0.1',port=a.port);server.scene.set_up_direction('+z');server.scene.add_grid('/floor',plane='xy')
server.gui.add_markdown('**SIMULATION ONLY · SONIC right-hand reach test**\n\nBlue: simulated robot links. Green: target. Orange: actual right-hand tracking point. The tracking point includes SONIC’s wrist offset, not a calibrated racket center. The table and racket are not modeled here.')
start=server.gui.add_button('Start right-hand reach');stop=server.gui.add_button('Stop simulation')
result_info=server.gui.add_markdown('First hold all three targets within 2 cm for one second. A completed command is not automatically a passed test.')
info=server.gui.add_html('<div style="height:160px">Waiting for simulation.</div>')
target=server.scene.add_icosphere('/target',radius=.025,color=(50,230,110),visible=False)
actual=server.scene.add_icosphere('/actual',radius=.02,color=(255,150,30),visible=False)
@start.on_click
def reach(_):
    if start.disabled:return
    tmp=a.run/'command.tmp';tmp.write_text(json.dumps({'id':time.time_ns(),'action':'reach'}));tmp.replace(a.run/'command.json')
@stop.on_click
def end(_):
    tmp=a.run/'command.tmp';tmp.write_text(json.dumps({'id':time.time_ns(),'action':'stop'}));tmp.replace(a.run/'command.json')
@server.on_client_connect
def connect(client):client.camera.position=(2,-2,1.7);client.camera.look_at=(0,0,.7)
while True:
    try:
        r=json.loads((a.run/'state.json').read_text());fresh=time.monotonic()-r['time']<.5 and not (a.run/'finished.json').exists()
        if (a.run/'finished.json').exists():r['phase']=json.loads((a.run/'finished.json').read_text())['phase']
        start.disabled=not fresh or not r.get('ready_for_reach',False);stop.disabled=not fresh
        pts=np.array(r['body_positions']);parents=r['parents'];pairs=np.array([[pts[i],pts[p]] for i,p in enumerate(parents) if i>1])
        server.scene.add_line_segments('/robot',points=pairs,colors=(90,160,230),line_width=7)
        actual.position=r['actual_world_m'][1];actual.visible=True
        if r['target_local_m'] is not None:
            target.position=np.array(r['root_position'])+np.array(r['root_rotation'])@np.array(r['target_local_m'][1]);target.visible=True
        error=r.get('right_error_cm');error_text='—' if error is None else f'{error:.2f} cm'
        info.content=f'<div style="height:160px">Phase: {r["phase"]}<br>Virtual support: {"ON (assisted test)" if r.get("virtual_support") else "OFF"}<br>Feed: {"Live simulation" if fresh else "Stopped / stale snapshot"}<br>Right-hand error: {error_text}<br>Root height: {r["root_height_m"]:.3f} m<br>Movement: {movement}, {ramp_seconds:g}-second ramp, {hold_seconds:g}-second hold.<br>No real-robot control.</div>'
        if (a.run/'finished.json').exists() and (a.run/'report.json').exists():
            report=json.loads((a.run/'report.json').read_text())
            if report.get('completed'):
                passed=report.get('short_reach_success',report.get('within_2cm_for_final_second',False))
                progress=report.get('movement_completion_ratio')
                progress_text='' if progress is None else f' Movement completed along target: {progress:.0%}.'
                result_info.content=f"**Saved result: {'PASS' if passed else 'NOT PASSED'}**\n\nStarting error: {report.get('starting_error_cm',0):.1f} cm. Final mean error: {report.get('mean_final_error_cm',0):.1f} cm.{progress_text} Support during test: {report.get('virtual_support')}."
            else:
                hold=report.get('baseline_hold_mean_errors_cm',report.get('final_hold_mean_errors_cm',{}))
                result_info.content=f"**Saved result: {report.get('exit_phase','NO REACH TEST')}**\n\nBaseline hold mean errors (cm): { {k:round(v,2) for k,v in hold.items()} }. Support during test: {report.get('virtual_support')}."

        server.flush()
    except FileNotFoundError:start.disabled=True
    time.sleep(.1)
