
import os, numpy as np, mujoco
try:
    import mujoco.viewer as mjv
except Exception:
    mjv=None

ASSET=os.path.join(os.path.dirname(__file__),"..","..","assets","dexbench_three.xml")

class DexBenchThreeFingerEnv:
    def __init__(self, render_mode=None, episode_length=1200):
        self.model=mujoco.MjModel.from_xml_path(ASSET); self.data=mujoco.MjData(self.model)
        self.render_mode=render_mode; self.episode_length=episode_length; self.t=0; self.viewer=None
    @property
    def action_space(self): return 6
    @property
    def observation_space(self): return 20
    def _obs(self,a):
        qpos=self.data.qpos.copy(); xyz=qpos[:3]; wxyz=qpos[3:7]
        quat_xyzw=np.array([wxyz[1],wxyz[2],wxyz[3],wxyz[0]],np.float32)
        jn=["a1","a2","b1","b2","c1","c2"]
        q=np.array([self.data.qpos[self.model.joint(n).id] for n in jn],np.float32)
        tfeat=np.array([self.t/self.episode_length],np.float32)
        return np.concatenate([a.astype(np.float32), xyz, quat_xyzw, q, tfeat])
    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        mujoco.mj_resetData(self.model,self.data); self.data.qpos[3:7]=np.array([1,0,0,0],np.float64); self.t=0
        return self._obs(np.zeros(6,np.float32)), {}
    def step(self, action):
        a=np.clip(np.asarray(action,np.float64),-1,1); self.data.ctrl[:]=a; mujoco.mj_step(self.model,self.data); self.t+=1
        wxyz=self.data.qpos[3:7]; tilt=1-wxyz[0]**2; r=-float(tilt)
        term=False; trunc=self.t>=self.episode_length; info={"quat_err": tilt}
        return self._obs(a.astype(np.float32)), r, term, trunc, info
    def render(self):
        if self.render_mode!="human" or mjv is None: return
        if self.viewer is None: self.viewer=mjv.launch_passive(self.model,self.data)
        self.viewer.sync()
    def close(self): pass
