import sys,numpy as np,cv2
sys.path.insert(0,'outputs/g1-ball-session')
from ball_center import refine_center
scale=16;y,x=np.mgrid[:64*scale,:64*scale];cx,cy=31.3,30.7
high=30+180*(((x+.5)/scale-.5-cx)**2+((y+.5)/scale-.5-cy)**2<5**2)
gray=cv2.resize(high.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA).astype(np.uint8)
fit=refine_center(gray,[26,25,37,36,.9,0]);assert fit and np.linalg.norm(np.array(fit['center_px'])-[cx,cy])<.2,fit
assert refine_center(np.full((64,64),100,np.uint8),[26,25,37,36,.9,0]) is None
assert refine_center(gray,[1,1,3,3,.9,0]) is None
assert refine_center(gray,[0,20,10,30,.9,0]) is None
print('PASS: subpixel synthetic circle, flat ROI, undersized and clipped ROI')
