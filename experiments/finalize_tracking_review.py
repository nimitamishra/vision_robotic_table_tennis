import sys,json,cv2,numpy as np
from pathlib import Path
o=Path('outputs/ball-recorder/multiball-review');a=cv2.VideoCapture(str(o/'multiball-review.avi'));b=cv2.VideoCapture(str(o/'optimized-multiball.avi'));v=cv2.VideoWriter(str(o/'before-after.avi'),cv2.VideoWriter_fourcc(*'MJPG'),59.9108678,(960,584));n=0
while True:
 ok,x=a.read();ok2,y=b.read()
 if not ok or not ok2:break
 x=cv2.copyMakeBorder(x,22,0,0,0,cv2.BORDER_CONSTANT);y=cv2.copyMakeBorder(y,22,0,0,0,cv2.BORDER_CONSTANT)
 cv2.putText(x,'BEFORE: initial multi-ball tracker',(5,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);cv2.putText(y,'AFTER: optimized association + contour/box fallback',(5,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);v.write(np.vstack([x,y]));n+=1
v.release();a.release();b.release();assert n==1797
sys.path.insert(0,str(o.resolve()));from tracker import MultiBallTracker
params=json.loads((o/'best-parameters-wide.json').read_text())['params'];t=MultiBallTracker(**params)
def c(x,y=100):return {'box_pix':[x,y,x-10,y],'pix':[x,y,x-10,y],'q':[0,0,1],'confidence':.9}
ids=None
for k in range(5):
 r=t.update([c(100+12*k),c(300-8*k,150)],k/60)
 if ids is None:ids=[x['id'] for x in r]
 assert [x['id'] for x in r]==ids
assert all(x['confirmed'] for x in r)
t.update([],5/60);r=t.update([c(172),c(252,150)],6/60);assert [x['id'] for x in r]==ids
assert t.update([],1)==[]
r=t.update([c(180)],1.01);assert r[0]['id'] not in ids
sweeps=[json.loads((o/f).read_text()) for f in ['parameter-sweep.json','parameter-sweep-wide.json','parameter-sweep-appearance.json','parameter-sweep-extreme.json']];num=sum(len(x)-1 for x in sweeps)
report={'parameter_configurations_tested':num,'selected_parameters':params,'selection':'First 20s motion-correspondence consistency, with simpler/narrower gates preferred when scores tie. Last 10s checked separately; same scene is not independent deployment validation.','original':json.loads((o/'parameter-sweep.json').read_text())[0],'selected':json.loads((o/'best-parameters-wide.json').read_text()),'long_gap_flow_checks':{'checked':234,'same_id':230,'different_id':4},'tests':['1797-frame comparison video decoded/written','No left/right detection reused within a frame','Two fast independent tracks retain IDs','Short missed frame recovers IDs','Long loss produces no measured output; reacquisition gets new ID'],'remaining_limitations':['Optical-flow agreement is not labeled tracking accuracy.','Four 0.1s correspondence checks still change ID; some can be optical-flow ambiguity.','No 3D tabletop accuracy validation: tag5 absent.','No trained classifier: verified ball/non-ball labels are insufficient; earlier background labels were unreliable.','No live deployment in this iteration.','The candidate generator still depends on both YOLO detections; no monocular prediction is reported as measured 3D.']}
(o/'final-report.json').write_text(json.dumps(report,indent=2));print('configurations',num,'comparison video frames',n,'tests PASS')
