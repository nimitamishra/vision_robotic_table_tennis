"""Bounded slow position feedback for MuJoCo experiments only."""
import numpy as np

def update_correction(correction,error,dt):
    correction=np.asarray(correction,float);error=np.asarray(error,float)
    if correction.shape!=(3,3) or error.shape!=(3,3) or not np.all(np.isfinite([correction,error])) or not np.isfinite(dt) or dt<0:
        raise ValueError('Finite three-point errors and nonnegative time required')
    updated=correction+np.clip(.5*error,-.01,.01)*min(.1,dt)
    norms=np.linalg.norm(updated,axis=1,keepdims=True)
    return updated*np.minimum(1.,.04/np.maximum(norms,1e-9))
