"""Front-facing table scene, provisional robot/racket geometry, simulation only."""
import mujoco
from impact_env import make_spec, CONFIG

TABLE = {'near_x': .85, 'length': 2.74, 'width': 1.525, 'height': .76,
         'thickness': .025, 'net_height': .1525, 'net_thickness': .008}
TABLE['net_x'] = TABLE['near_x']+TABLE['length']/2
TABLE['far_x'] = TABLE['near_x']+TABLE['length']

def make_front_model():
    s = make_spec()
    s.option.timestep = .001
    s.pair("ball_face_contact").solref = [.016, .1]
    s.worldbody.add_geom(name='table_top', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[TABLE['net_x'], 0, TABLE['height']-TABLE['thickness']/2],
        size=[TABLE['length']/2, TABLE['width']/2, TABLE['thickness']/2],
        rgba=[.07, .24, .42, 1])
    s.worldbody.add_geom(name='table_net', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[TABLE['net_x'], 0, TABLE['height']+TABLE['net_height']/2],
        size=[TABLE['net_thickness']/2, TABLE['width']/2+.03, TABLE['net_height']/2],
        rgba=[.8, .85, .9, .55])
    for x in (TABLE['near_x']+.25, TABLE['far_x']-.25):
        for y in (-.60, .60):
            s.worldbody.add_geom(name=f'table_leg_{x}_{y}', type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, .36], size=[.025, .025, .36], rgba=[.2, .2, .2, 1])
    s.add_pair(name='ball_table_contact', geomname1='table_top', geomname2='training_ball_geom',
        condim=3, solref=[.016, .1], solimp=CONFIG['solimp'], friction=[.1, .1, .001, .0001, .0001])
    return s.compile()
