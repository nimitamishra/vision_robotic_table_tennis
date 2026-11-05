"""Live/replay single-ball trajectory estimation; no robot control dependencies."""
import argparse,json,time,signal,sys,queue
from html import escape
from pathlib import Path
from collections import deque
import numpy as np
import zmq,msgpack
from tracker import Tracker,measurement,G
from collision import forecast as collision_forecast
from intercept_session import Session

def fixed_panel(text, height):
    """Keep live telemetry from moving controls when text wraps or disappears."""
    return (f'<div style="height:{height}px;overflow:auto;white-space:pre-wrap;'
            'overflow-wrap:anywhere;box-sizing:border-box;font-variant-numeric:tabular-nums">'
            + escape(text.replace("**", "")) + "</div>")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='tcp://127.0.0.1:5563')
    ap.add_argument('--publish',default='tcp://127.0.0.1:5564')
    ap.add_argument('--output',type=Path,default=Path(__file__).parent/'runs')
    ap.add_argument('--replay',type=Path,help='Replay measurements.jsonl recorded by this script')
    ap.add_argument('--demo',action='store_true',help='Synthetic ballistic data, not real observations')
    ap.add_argument('--viser-port',type=int,default=8082)
    ap.add_argument('--no-viewer',action='store_true')
    ap.add_argument('--horizon',type=float,default=.5,help='Maximum free-flight forecast in seconds')
    ap.add_argument('--max-gap',type=float,default=.2)
    ap.add_argument('--contour-sigma',type=float,default=.012)
    ap.add_argument('--box-sigma',type=float,default=.025)
    ap.add_argument('--restitution',type=float,default=.8)
    ap.add_argument('--tangent-retention',type=float,default=.9)
    ap.add_argument('--table-z',type=float,default=0.)
    ap.add_argument('--g1-dir',type=Path,default=Path('/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1'),help='Shared table_scene.py and calibration.json used by Viser')
    a=ap.parse_args()
    if not .05<=a.horizon<=1.:ap.error('Horizon must be between 0.05 and 1 second')
    if a.max_gap<=0 or a.contour_sigma<=0 or a.box_sigma<=0:ap.error('Gap and noise scales must be positive')
    if not 0<=a.restitution<=1 or not 0<=a.tangent_retention<=1:ap.error('Impact parameters must be between 0 and 1')
    if a.demo and a.replay:ap.error('Choose demo or replay, not both')
    out=a.output/time.strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
    calibration=json.loads((a.g1_dir/'calibration.json').read_text());table_spec=calibration['world']['table']
    intercept=Session(out);test_commands=queue.Queue();last_test_ui=0.
    impact={'enabled':True,'restitution':a.restitution,'retention':a.tangent_retention}
    tracker=Tracker(a.max_gap,a.contour_sigma,a.box_sigma,a.table_z,horizon=a.horizon)
    ctx=zmq.Context();pub=ctx.socket(zmq.PUB);pub.setsockopt(zmq.SNDHWM,1);pub.bind(a.publish)
    sub=None;stop=False;server=None;trail=deque(maxlen=60);last_epoch=None;last_packet=time.monotonic();stale_sent=False
    def end(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,end);signal.signal(signal.SIGTERM,end)
    if not a.no_viewer:
        import viser
        server=viser.ViserServer(host='127.0.0.1',port=a.viser_port)
        server.scene.set_up_direction('+z')
        sys.path.insert(0,str(a.g1_dir))
        from table_scene import add_table
        calibration=json.loads((a.g1_dir/'calibration.json').read_text())
        table_spec=calibration['world']['table']
        (out/'scene-calibration.json').write_text(json.dumps(calibration,indent=2))
        table_root,_=add_table(server.scene,table_spec)
        table_root.position=(0,0,a.table_z)
        center_y=table_spec['length_m']/2-table_spec['tag_center_inset_m']
        server.scene.add_grid('/floor',plane='xy',position=(0,center_y,a.table_z-table_spec['height_m']))
        server.scene.add_frame('/table_origin',axes_length=.12,axes_radius=.003,position=(0,0,a.table_z))
        server.scene.add_label('/table_origin/label','Table origin (Tag 5)',position=(0,0,.08))
        horizon_control=server.gui.add_slider('Forecast horizon (s)',min=.05,max=1.,step=.05,initial_value=a.horizon)
        @horizon_control.on_update
        def change_horizon(_):tracker.horizon=float(horizon_control.value)
        collision_toggle=server.gui.add_checkbox('Collision forecast',True)
        @collision_toggle.on_update
        def enable_collision(_):impact['enabled']=collision_toggle.value
        restitution_control=server.gui.add_slider('Restitution (uncalibrated)',min=0.,max=1.,step=.05,initial_value=a.restitution)
        @restitution_control.on_update
        def change_restitution(_):impact['restitution']=float(restitution_control.value)
        retention_control=server.gui.add_slider('Horizontal retention',min=0.,max=1.,step=.05,initial_value=a.tangent_retention)
        @retention_control.on_update
        def change_retention(_):impact['retention']=float(retention_control.value)
        session_fit=server.gui.add_button('Try session impact fit (0.90 / 0.60)')
        server.gui.add_markdown('Experimental fit from 3 impacts on Oct 30; not a universal ball/table calibration.')
        @session_fit.on_click
        def try_session_fit(_):
            restitution_control.value=.9;retention_control.value=.6
            impact['restitution']=.9;impact['retention']=.6
        collision_marker=server.scene.add_icosphere('/collision',radius=.03,color=(255,75,75),visible=False)
        collision_info=server.gui.add_html(fixed_panel('Collision: waiting for velocity estimate.', 150))
        show_table=server.gui.add_checkbox('Show table',True)
        @show_table.on_update
        def toggle_table(_):table_root.visible=show_table.value
        with server.gui.add_folder('Intercept Test'):
            plane_control=server.gui.add_slider('Plane from near edge (m)',min=0.,max=.6,step=.01,initial_value=.2)
            count_control=server.gui.add_slider('Number of throws',min=1,max=50,step=1,initial_value=20)
            start_test=server.gui.add_button('Start Intercept Test')
            stop_test=server.gui.add_button('Stop & Save Test')
            test_info=server.gui.add_html(fixed_panel('Fixed robot. One ball. Start the test, then throw only during THROW ONE BALL. No SONIC required.', 220))
            server.gui.add_markdown('100/200/300 ms errors use later camera observations, not independent ground truth. Cancelled and missed trials are recorded.')
        test_plane=server.scene.add_box('/intercept_plane',dimensions=(table_spec['width_m'],.003,1.18),
            position=(0,-table_spec['tag_center_inset_m']+.2,a.table_z+.61),color=(65,210,165),opacity=.15,visible=False)
        intercept_marker=server.scene.add_icosphere('/intercept_prediction',radius=.025,color=(65,240,150),visible=False)
        @start_test.on_click
        def start_intercept(_):test_commands.put(('start',float(plane_control.value),int(count_control.value)))
        @stop_test.on_click
        def stop_intercept(_):test_commands.put(('stop',))
        raw_ball=server.scene.add_icosphere('/raw_ball',radius=.012,color=(170,170,170),visible=False)
        ball=server.scene.add_icosphere('/ball',radius=.02,color=(80,190,240),visible=False)
        info=server.gui.add_html(fixed_panel('Waiting for table-frame stereo measurements.', 400))
        server.gui.add_markdown('Acceleration = gravity model, not a measured signal. Collision model: finite table, one estimated rebound; net/edge risks stop the forecast. Impact parameters are uncalibrated.\n\n'+('**SYNTHETIC DEMO**' if a.demo else '**REPLAY**' if a.replay else '**LIVE · no motion commands**'))
        @server.on_client_connect
        def view(client):client.camera.position=(2.7,-2.4,a.table_z+2.3);client.camera.look_at=(0,center_y,a.table_z+.15)
    def service_test():
        nonlocal last_test_ui
        now=time.monotonic()
        while not test_commands.empty():
            cmd=test_commands.get_nowait()
            if cmd[0]=='start' and not intercept.active:
                intercept.start(now,-table_spec['tag_center_inset_m']+cmd[1],table_spec['width_m'],a.table_z,cmd[2])
                if server:test_plane.position=(0,intercept.plane_y,a.table_z+.61);test_plane.visible=True
            elif cmd[0]=='stop':intercept.stop()
        intercept.tick(now)
        if not a.demo and not a.replay and now-last_packet>a.max_gap:intercept.current_prediction=None
        if server and now-last_test_ui>=.1:
            test_text=intercept.label(now)
            start_test.disabled=intercept.active;plane_control.disabled=intercept.active;count_control.disabled=intercept.active
            pred=intercept.current_prediction;intercept_marker.visible=pred is not None
            if pred is not None:
                intercept_marker.position=pred['position_m']
                test_text+=f"\n\nPredicted crossing: {pred['time_s']*1000:.0f} ms after input frame; {np.round(pred['position_m'],3).tolist()} m."
            if intercept.results:
                recent=intercept.results[-1];test_text+=f"\n\nLast trial: {recent['status']}"
                for metric in recent['lead_metrics']:
                    if metric['status']=='evaluated':test_text+=f"\n\n{metric['requested_lead_s']*1000:.0f} ms lead: {metric['position_error_m']*100:.1f} cm, {metric['time_error_s']*1000:+.0f} ms"
            test_info.content=fixed_panel(test_text, 220)
            last_test_ui=now
    history_segments=deque(maxlen=120);previous_history=None;last_draw=0.;last_snapshot=0.
    def draw(result):
        nonlocal last_epoch,previous_history,last_draw
        if not server:return
        if time.monotonic()-last_draw<1/30:return
        last_draw=time.monotonic()
        pos=result.get('position_m');ball.visible=pos is not None
        now=time.monotonic();epoch=result.get('track_epoch')
        if pos is not None:
            ball.position=pos
            if previous_history is not None and previous_history[0]==epoch:
                history_segments.append((now,[previous_history[1],pos]))
            previous_history=(epoch,pos)
        else:previous_history=None
        while history_segments and now-history_segments[0][0]>1.:history_segments.popleft()
        segments=[q[1] for q in history_segments]
        server.scene.add_line_segments('/history',points=np.asarray(segments if segments else [[[0,0,0],[0,0,0]]]),colors=(80,190,240),line_width=3,visible=bool(segments))
        raw_pos=result.get('raw_position_m');raw_ball.visible=raw_pos is not None
        if raw_pos is not None:raw_ball.position=raw_pos
        def lines(name,points,color):
            arr=np.asarray(points,dtype=float)
            if len(arr)<2:arr=np.zeros((2,3))
            server.scene.add_line_segments(name,points=np.stack([arr[:-1],arr[1:]],axis=1),colors=color,line_width=3,visible=len(points)>1)
        predictions=result.get('predictions',[])
        collision_segments=result.get('collision_segments')
        if collision_segments is None:
            lines('/forecast',result.get('forecast_path',[]),(255,180,65));lines('/rebound',[],(190,100,245))
        else:
            lines('/forecast',next((seg['points'] for seg in collision_segments if seg['phase']=='free_flight'),[]),(255,180,65))
            lines('/rebound',next((seg['points'] for seg in collision_segments if seg['phase']=='after_bounce'),[]),(190,100,245))
        events=result.get('collision_events',[]);collision_marker.visible=bool(events)
        if events:
            first=events[0];collision_marker.position=first['ball_center_m']
            contact_text=f"**Predicted contact:** {first['type']} in {first['time_s']*1000:.0f} ms\n\n**Center (m):** {np.round(first['ball_center_m'],3).tolist()}\n\nPurple: estimated rebound. Red: first contact. Parameters are not calibrated."
        else:contact_text='No predicted contact within horizon.' if collision_segments else 'Collision: waiting for a flight estimate.'
        collision_info.content=fixed_panel(contact_text, 150)
        vel=result.get('velocity_m_s');speed=result.get('speed_m_s')
        status_text=f"**Status:** {result['status']}\n\n**Velocity (m/s):** {np.round(vel,3).tolist() if vel else '—'}\n\n**Speed:** {f'{speed:.3f} m/s' if speed is not None else '—'}\n\n**Event:** {result.get('event') or '—'}\n\n**Source:** {result.get('source','—')}\n\n**Model acceleration (m/s²):** {result.get('acceleration_m_s2','—')}\n\n**Forecast length:** {round(result.get('forecast_duration_s',0)*1000)} ms ({result.get('forecast_limit','not ready')})\n\n**Input:** {result.get('input_status','no new data')}\n\n**Warm-up samples:** {result.get('initialization_samples','ready')}\n\nBlue: recent history (1s). Orange: model forecast. Gray: raw 3D point. Longer forecasts are less certain.\n\n**Desktop input age:** {round(result.get('desktop_pipeline_age_s',0)*1000)} ms (excludes camera/network and browser rendering)"
        info.content=fixed_panel(status_text, 400)
        server.flush()
    if a.replay:
        replay=iter(json.loads(x) for x in a.replay.read_text().splitlines() if x.strip())
    elif not a.demo:
        sub=ctx.socket(zmq.SUB);sub.setsockopt(zmq.SUBSCRIBE,b'');sub.setsockopt(zmq.CONFLATE,1);sub.connect(a.input)
    count=0;rng=np.random.default_rng(4);start=time.monotonic()
    print(f"Trajectory ready. Output: {out}\nInput: {a.input}; Viewer: http://127.0.0.1:{a.viser_port}",flush=True)
    try:
        with (out/'measurements.jsonl').open('w') as raw,(out/'trajectory.jsonl').open('w') as output:
            while not stop:
                service_test()
                if a.replay:
                    try:packet=next(replay)
                    except StopIteration:break
                    time.sleep(1/60)
                elif a.demo:
                    phase=count%60;t=phase/60
                    pos=np.array([-.3,.2,1.2])+np.array([.5,1.,1.])*t+.5*G*t*t
                    packet={'capture_monotonic_ns':int((count/60+1)*1e9),'frame_id':count,'world_measurement':True,'ball':{},'xyz_world':(pos+rng.normal(0,.005,3)).tolist() if pos[2]>.02 else None}
                    count+=1;time.sleep(1/60)
                else:
                    if not sub.poll(100):
                        if time.monotonic()-last_packet>a.max_gap and stale_sent:
                            draw({'status':'stream_stale','predictions':[],'event':'no_new_packets'})
                        if time.monotonic()-last_packet>a.max_gap and not stale_sent:
                            result={'status':'stream_stale','predictions':[],'event':'no_new_packets'};draw(result);pub.send(msgpack.packb(result,use_bin_type=True));output.write(json.dumps(result)+'\n');output.flush()
                            (out/'latest.json').write_text(json.dumps(result));stale_sent=True
                        continue
                    packet=msgpack.unpackb(sub.recv(),raw=False);last_packet=time.monotonic();stale_sent=False
                stamp=packet.get('capture_monotonic_ns')
                if not isinstance(stamp,(int,float)) or not np.isfinite(stamp):
                    result={'status':'missing_capture_timestamp','predictions':[]}
                else:
                    p,source=measurement(packet);result=tracker.step(stamp*1e-9,p,source)
                    result['raw_position_m']=p.tolist() if p is not None else None
                    result['input_status']=packet.get('measurement_status') or ('valid_3d' if p is not None else 'no_valid_3d')
                    result['requested_horizon_s']=tracker.horizon
                    result['impact_settings']=dict(impact)
                    if impact['enabled'] and result.get('velocity_m_s') is not None and tracker.mode!='stationary':
                        result.update(collision_forecast(result['position_m'],result['velocity_m_s'],table_spec,
                            horizon=tracker.horizon,table_z=a.table_z,restitution=impact['restitution'],tangent_retention=impact['retention']))
                        result['forecast_duration_s']=result['collision_forecast_duration_s']
                        result['forecast_limit']=result['collision_forecast_limit']
                    boxes=packet.get('boxes')
                    if p is None and boxes is not None:
                        left,right=map(len,boxes)
                        result['input_status']+=f' | Left candidates: {left}; Right candidates: {right}'
                    result['prediction_reference']='robot frame-receipt timestamp; not latency-compensated to current wall time'
                    if packet.get('source_received_monotonic') is not None:
                        result['desktop_pipeline_age_s']=max(0.,time.monotonic()-packet['source_received_monotonic'])
                    intercept.observe(time.monotonic(),stamp*1e-9,p,result)
                    result['intercept_test']={'active':intercept.active,'phase':intercept.phase,'prediction':intercept.current_prediction}
                    result['frame_id']=packet.get('frame_id');result['mode']='synthetic' if a.demo else 'replay' if a.replay else 'live'
                slim={k:packet.get(k) for k in ('capture_monotonic_ns','frame_id','world_measurement','ball','ball_refined','xyz_world','xyz_world_refined','measurement_status','boxes','tag_ids')}
                pub.send(msgpack.packb(result,use_bin_type=True));draw(result)
                raw.write(json.dumps(slim)+'\n');output.write(json.dumps(result)+'\n')
                if time.monotonic()-last_snapshot>=.1:
                    raw.flush();output.flush()
                    temp=out/'latest.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(out/'latest.json');last_snapshot=time.monotonic()
                if time.monotonic()-start>1:print(f"{result['status']} | speed={result.get('speed_m_s')} | {result.get('event')}",flush=True);start=time.monotonic()
    finally:
        intercept.stop()
        if sub is not None:sub.close(0)
        pub.close(0);ctx.term()
        if server:server.stop()

if __name__=='__main__':main()
