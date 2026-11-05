"""Offline multi-ball association in stereo pixel coordinates; no world-space gate."""
import numpy as np
from scipy.optimize import linear_sum_assignment

class MultiBallTracker:
    def __init__(self, max_gap=.15, initial_gate=35, residual_gate=12, velocity_weight=.8, age_penalty=0., appearance_weight=0.):
        self.tracks=[];self.next_id=1;self.max_gap=max_gap
        self.age_penalty=age_penalty;self.appearance_weight=appearance_weight
        self.initial_gate=initial_gate;self.residual_gate=residual_gate;self.velocity_weight=velocity_weight

    def update(self, candidates, t):
        self.tracks=[tr for tr in self.tracks if t-tr['t']<=self.max_gap]
        # Stereo assignment with explicit unmatched choices; a detection cannot be reused.
        left=list(dict.fromkeys(tuple(c['box_pix'][:2]) for c in candidates))
        right=list(dict.fromkeys(tuple(c['box_pix'][2:]) for c in candidates))
        chosen=[]
        if left and right:
            cost=np.full((len(left),len(right)+len(left)),2.5);lookup={}
            for c in candidates:
                i=left.index(tuple(c['box_pix'][:2]));j=right.index(tuple(c['box_pix'][2:]));p=np.array(c['box_pix'])
                affinity=min([np.linalg.norm(p-(tr['p']+tr['v']*(t-tr['t'])))/30 for tr in self.tracks]+[1.])
                value=abs(p[1]-p[3])+.3*(1-c['confidence'])+.3*min(1,affinity)
                if value<cost[i,j]:cost[i,j]=value;lookup[i,j]=c
            rr,cc=linear_sum_assignment(cost)
            chosen=[lookup[i,j] for i,j in zip(rr,cc) if (i,j) in lookup]
        matched={};used=set()
        if self.tracks and chosen:
            costs=np.full((len(self.tracks),len(chosen)+len(self.tracks)),1.1)
            for i,tr in enumerate(self.tracks):
                dt=t-tr['t'];pred=tr['p']+tr['v']*dt;gate=(self.initial_gate if tr['hits']<3 else self.residual_gate)+100*dt
                for j,c in enumerate(chosen):
                    pix=np.array(c['box_pix']);e=np.linalg.norm(pix-pred)/np.sqrt(2)
                    if e<=gate and np.linalg.norm(pix-tr['p'])/np.sqrt(2)<=8+1800*dt:costs[i,j]=e/gate+self.age_penalty*max(0,dt-.02)/self.max_gap
                    if costs[i,j]<1.1 and self.appearance_weight and 'feat' in c and 'feat' in tr:
                        costs[i,j]+=self.appearance_weight*min(1,float(np.mean((np.array(c['feat'])-tr['feat'])**2))/2)
            rr,cc=linear_sum_assignment(costs)
            for i,j in zip(rr,cc):
                if j<len(chosen) and costs[i,j]<1.1:matched[i]=j;used.add(j)
        output=[]
        for i,tr in enumerate(self.tracks):
            if i not in matched:continue
            c=chosen[matched[i]];p=np.array(c['box_pix']);dt=max(.001,t-tr['t']);weight=1. if tr['hits']==1 else self.velocity_weight
            tr['v']=(1-weight)*tr['v']+weight*(p-tr['p'])/dt
            tr.update(p=p,t=t,hits=tr['hits']+1)
            if 'feat' in c:tr['feat']=np.array(c['feat'])
            output.append(dict(id=tr['id'],confirmed=tr['hits']>=3,**c))
        for j,c in enumerate(chosen):
            if j in used:continue
            tr={'id':self.next_id,'p':np.array(c['box_pix']),'v':np.zeros(4),'t':t,'hits':1};self.next_id+=1;self.tracks.append(tr)
            if 'feat' in c:tr['feat']=np.array(c['feat'])
            output.append(dict(id=tr['id'],confirmed=False,**c))
        return output
