"""Ball-state observation proxy for reach training; no collision or bounce physics."""
import numpy as np
from fixed_arm_env import FixedArmReach
G=np.array([0.,0.,-9.81])
class BallProxyReach(FixedArmReach):
    def __init__(self,seed=0):
        self.ball_v0=np.zeros(3)
        self.ball_p0=np.zeros(3)
        super().__init__(seed=seed,steps=10)
    def reset(self,goal=None):
        # Choose a ball that crosses a reachable target after 0.5 s.
        self.ball_v0=np.array([self.rng.uniform(-1.5,-.6),self.rng.uniform(-.15,.15),self.rng.uniform(1.2,2.2)])
        super().reset(goal)
        horizon=self.steps*.05
        self.ball_p0=self.goal-self.ball_v0*horizon-.5*G*horizon*horizon
        return self.observe()
    def ball(self):
        t=self.t*.05
        return self.ball_p0+self.ball_v0*t+.5*G*t*t,self.ball_v0+G*t
    def observe(self):
        p,v=self.ball()
        remaining=max(0,self.steps-self.t)*.05
        crossing=p+v*remaining+.5*G*remaining*remaining
        return np.r_[self.data.qpos,self.data.qvel,self.point()-self.ready,p-self.ready,v,
                     remaining,crossing-self.ready].astype(np.float32)
    def step(self,action):
        _,reward,done,info=super().step(action)
        # Final proxy error is the wrist-to-ball distance at the chosen crossing.
        if done:
            info['proxy_contact_error_m']=float(np.linalg.norm(self.point()-self.ball()[0]))
            reward-=40*info['proxy_contact_error_m']
        return self.observe(),reward,done,info
