"""Local intensity-contour ellipse fit; no temporal filtering or learned weights."""
import cv2
import numpy as np


def refine_center(gray, box):
    x1,y1,x2,y2 = map(float,box[:4]); w=x2-x1; h=y2-y1
    if min(w,h)<4 or max(w,h)>80: return None
    pad=max(3,int(max(w,h)*.45))
    x0=max(0,int(np.floor(x1))-pad);y0=max(0,int(np.floor(y1))-pad)
    xe=min(gray.shape[1],int(np.ceil(x2))+pad);ye=min(gray.shape[0],int(np.ceil(y2))+pad)
    if x0==0 or y0==0 or xe==gray.shape[1] or ye==gray.shape[0]: return None
    roi=gray[y0:ye,x0:xe].astype(np.float32)
    bg=float(np.percentile(roi,35)); bright=float(np.percentile(roi,95))
    if bright-bg<25: return None
    scale=4
    up=cv2.resize(roi,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    expected=np.array([(x1+x2)/2-x0,(y1+y2)/2-y0])
    fits=[]
    for fraction in (.35,.5,.65):
        mask=(up>bg+fraction*(bright-bg)).astype(np.uint8)*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        choices=[]
        for contour in contours:
            if len(contour)<12: continue
            (cx,cy),(a,b),angle=cv2.fitEllipse(contour)
            # OpenCV resize pixel-center convention.
            center=(np.array([cx,cy])+.5)/scale-.5
            a/=scale;b/=scale
            if min(a,b)<3 or max(a,b)>max(w,h)*1.6 or min(a,b)/max(a,b)<.65: continue
            if np.linalg.norm(center-expected)>max(2,min(w,h)*.35): continue
            area=cv2.contourArea(contour)/scale**2
            fill=area/(np.pi*a*b/4)
            if not .8<fill<1.15: continue
            if np.any(contour[:,0,0]<=0) or np.any(contour[:,0,1]<=0) or np.any(contour[:,0,0]>=up.shape[1]-1) or np.any(contour[:,0,1]>=up.shape[0]-1): continue
            choices.append((np.linalg.norm(center-expected),center, a,b))
        if choices: fits.append(min(choices,key=lambda c:c[0]))
    if len(fits)<3: return None
    centers=np.array([v[1] for v in fits]);center=np.median(centers,axis=0)
    spread=float(np.max(np.linalg.norm(centers-center,axis=1)))
    if spread>.6: return None
    return {'center_px':(center+[x0,y0]).tolist(),'threshold_spread_px':spread,'method':'local_isophote_ellipse'}


def refine_boxes(gray,boxes):
    refined=[];diagnostics=[]
    for box in boxes:
        fit=refine_center(gray,box)
        if fit is None: continue
        cx,cy=fit['center_px'];w=box[2]-box[0];h=box[3]-box[1]
        refined.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2,*box[4:]])
        diagnostics.append(fit)
    return refined,diagnostics
