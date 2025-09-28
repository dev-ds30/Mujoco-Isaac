
import argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from dexbench.dexbench_env_three import DexBenchThreeFingerEnv
from dexbench.dexbench_env import DexBenchEnv
class R(nn.Module):
    def __init__(self,din): super().__init__(); self.m=nn.Sequential(nn.Linear(din,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
    def forward(self,x): return self.m(x).squeeze(-1)
class Actor(nn.Module):
    def __init__(self,din,dout): super().__init__(); self.m=nn.Sequential(nn.Linear(din,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,dout),nn.Tanh())
    def forward(self,x): return self.m(x)
def rollout(env,pi,T,rm):
    obs,_=env.reset(); traj={'o':[],'a':[],'r':[]}
    for t in range(T):
        a=pi(torch.from_numpy(obs).float()).detach().numpy()
        obs2,_,term,trunc,_=env.step(a); r=rm(torch.from_numpy(obs).float()).item()
        traj['o'].append(obs); traj['a'].append(a); traj['r'].append(r); obs=obs2
        if term or trunc: break
    import numpy as _np
    for k in traj: traj[k]=_np.array(traj[k]); return traj
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--env',choices=['two','three'],default='three'); ap.add_argument('--rm',required=True); ap.add_argument('--total-steps',type=int,default=100000); a=ap.parse_args()
    blob=torch.load(a.rm,map_location='cpu'); rm=R(blob['din']); rm.load_state_dict(blob['model']); rm.eval()
    env=DexBenchThreeFingerEnv() if a.env=='three' else DexBenchEnv(); din, dout=env.observation_space, env.action_space; pi=Actor(din,dout); opt=optim.Adam(pi.parameters(), lr=1e-4)
    steps=0
    while steps<a.total_steps:
        tr=rollout(env,pi,1024,rm); obs=torch.from_numpy(tr['o']).float(); act=torch.from_numpy(tr['a']).float(); rew=torch.from_numpy(tr['r']).float()
        adv=(rew - rew.mean())/(rew.std()+1e-6); pred=pi(obs); loss=((pred-act)**2 * adv.unsqueeze(-1).clamp(min=0)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); steps+=len(obs); print('steps',steps,'loss',float(loss))
    print('done')
