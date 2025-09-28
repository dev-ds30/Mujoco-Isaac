
import argparse, os, time, numpy as np, mujoco, mujoco.viewer
from dexbench.dexbench_env_three import DexBenchThreeFingerEnv
from dexbench.dexbench_env import DexBenchEnv

KEYS3=[('q',0,+0.6),('a',0,-0.6),('w',1,+0.6),('s',1,-0.6),('e',2,+0.6),('d',2,-0.6),('r',3,+0.6),('f',3,-0.6),('t',4,+0.6),('g',4,-0.6),('y',5,+0.6),('h',5,-0.6)]
KEYS2=[('q',0,+0.6),('a',0,-0.6),('w',1,+0.6),('s',1,-0.6),('e',2,+0.6),('d',2,-0.6),('r',3,+0.6),('f',3,-0.6)]

def teleop(env_name='three', out_dir='demos/teleop', episode_length=1200, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    env = DexBenchThreeFingerEnv(render_mode='human', episode_length=episode_length) if env_name=='three' else DexBenchEnv(render_mode='human', episode_length=episode_length)
    act = np.zeros(env.action_space, np.float32); keys = KEYS3 if env.action_space==6 else KEYS2
    obs,_=env.reset(seed=seed); pressed=set(); data={'observations':[],'actions':[],'rewards':[],'infos':[]}

    def on_key(key, action):
        try: k=chr(key).lower()
        except: return
        if action==mujoco.viewer._KEY_PRESS: pressed.add(k)
        elif action==mujoco.viewer._KEY_RELEASE: pressed.discard(k)

    viewer=mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key)
    for t in range(episode_length):
        delta=np.zeros_like(act)
        for k,i,v in keys:
            if k in pressed: delta[i]+=v
        act=0.9*act+0.1*np.clip(delta,-1,1)
        obs,r,term,trunc,info=env.step(act); env.render()
        data['observations'].append(obs.tolist()); data['actions'].append(act.tolist()); data['rewards'].append(float(r)); data['infos'].append(info)
        if term or trunc: break

    fn=os.path.join(out_dir,f'{env_name}_{int(time.time())}.npz')
    np.savez_compressed(fn, **{k:np.array(v, dtype=object if k=='infos' else None) for k,v in data.items()})
    print('Saved', fn)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--env',choices=['two','three'],default='three'); ap.add_argument('--out',default='demos/teleop'); ap.add_argument('--len',type=int,default=1200); ap.add_argument('--seed',type=int,default=0); a=ap.parse_args(); teleop(a.env,a.out,a.len,a.seed)
