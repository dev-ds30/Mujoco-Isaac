
import asyncio, json, os, time, numpy as np, websockets
from dexbench.dexbench_env_three import DexBenchThreeFingerEnv
from dexbench.dexbench_env import DexBenchEnv
class Teleop:
    def __init__(self, env='three', episode_length=1200, out='demos/teleop_web'):
        os.makedirs(out, exist_ok=True); self.out=out
        self.env = DexBenchThreeFingerEnv(render_mode='human', episode_length=episode_length) if env=='three' else DexBenchEnv(render_mode='human', episode_length=episode_length)
        self.action_dim = self.env.action_space; self.reset()
    def reset(self):
        self.obs,_=self.env.reset(); self.t=0; self.buf={'observations':[],'actions':[],'rewards':[],'infos':[]}
    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a); self.env.render(); self.t+=1
        self.buf['observations'].append(obs.tolist()); self.buf['actions'].append(a.tolist()); self.buf['rewards'].append(float(r)); self.buf['infos'].append(info)
        return term or trunc
    def save(self, note='web'):
        fn=os.path.join(self.out, f"web_{int(time.time())}.npz"); np.savez_compressed(fn, **{k:np.array(v, dtype=object if k=='infos' else None) for k,v in self.buf.items()}, note=note); return fn
async def handler(ws):
    rec=None
    async for msg in ws:
        d=json.loads(msg)
        if d['cmd']=='init':
            rec=Teleop(env=d.get('env','three'), episode_length=int(d.get('episode_length',1200))); await ws.send(json.dumps({'status':'ready','action_dim':rec.action_dim}))
        elif d['cmd']=='action' and rec is not None:
            import numpy as _np
            a=_np.array(d['a'], _np.float32); done=rec.step(a); await ws.send(json.dumps({'t':rec.t,'done':done})); 
            if done: path=rec.save(d.get('note','web')); await ws.send(json.dumps({'saved':path})); rec.reset()
        elif d['cmd']=='stop' and rec is not None:
            path=rec.save(d.get('note','web')); await ws.send(json.dumps({'saved':path,'done':True})); rec.reset()
async def main():
    port=int(os.environ.get('DEX_WS_PORT','8765')); async with websockets.serve(handler, '0.0.0.0', port): print('WS on ws://localhost:%d' % port); await asyncio.Future()
if __name__=='__main__': asyncio.run(main())
