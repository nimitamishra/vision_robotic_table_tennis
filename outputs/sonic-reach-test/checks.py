"""Readiness rules for the simulation experiment; not a real-robot safety system."""
import math

def settled(history,now,window=1.,tolerance_cm=2.):
    recent=[(t,e) for t,e in history if now-window-.1<=t<=now]
    return (len(recent)>=15 and recent[0][0]<=now-window+.05
            and recent[-1][0]>=now-.1
            and all(math.isfinite(e) and 0<=e<=tolerance_cm for t,e in recent))
