
import argparse, numpy as np, torch
from dexbench.dexbench_env_three import DexBenchThreeFingerEnv
from dexbench.dexbench_env import DexBenchEnv
class Policy(torch.nn.Module):
    def __init__(self, ckpt):
        blob=torch.load(ckpt,map_location='cpu'); self.din,self.dout=blob['din'],blob['dout']
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(self.din,256),torch.nn.ReLU(),torch.nn.Linear(256,256),torch.nn.ReLU(),torch.nn.Linear(256,self.dout)); self.net.load_state_dict(blob['model']); self.eval()
    def act(self,obs): 
        with torch.no_grad(): return torch.tanh(self.net(torch.from_numpy(obs).float())).numpy()
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--env',choices=['two','three'],default='three'); ap.add_argument('--model',required=True); ap.add_argument('--steps',type=int,default=1200); ap.add_argument('--render',action='store_true'); a=ap.parse_args()
    env=DexBenchThreeFingerEnv(render_mode='human',episode_length=a.steps) if a.env=='three' else DexBenchEnv(render_mode='human',episode_length=a.steps)
    pi=Policy(a.model); obs,_=env.reset(); ret=0.0
    for t in range(a.steps):
        act=np.clip(pi.act(obs),-1,1)[0]; obs,r,term,trunc,info=env.step(act); ret+=r; 
        if a.render: env.render()
        if term or trunc: break
    print('return', ret)
