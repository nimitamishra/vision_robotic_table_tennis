import json
from common import *
if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True)
    m=next(m for m in motions() if m['name']=='table_tennis_forehand_y+000_z000')
    for stretch in (1.,1.5,2.):
        frames,report=rollout(m,stretch=stretch)
        (RUN/f'teacher-{stretch}.json').write_text(json.dumps({'report':report,'frames':frames}))
        print(json.dumps(report),flush=True)
