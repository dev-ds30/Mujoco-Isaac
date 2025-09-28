
import argparse, json, numpy as np, torch, torch.nn as nn, torch.optim as optim, os
class R(nn.Module):
    def __init__(self,din): super().__init__(); self.m=nn.Sequential(nn.Linear(din,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
    def forward(self,x): return self.m(x).squeeze(-1)
def load_seg(p):
    d=np.load(p['file'],allow_pickle=True); return d['observations'][p['start']:p['end']].astype(np.float32)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',required=True); ap.add_argument('--labels',required=True); ap.add_argument('--out',default='runs/reward.pt'); a=ap.parse_args()
    pairs=json.load(open(a.pairs))['pairs']; labels={int(x['idx']):x['label'] for x in json.load(open(a.labels))['labels']}
    din=load_seg(pairs[0]['a']).shape[1]; model=R(din); opt=optim.AdamW(model.parameters(),lr=3e-4)
    for ep in range(10):
        L=0.0; n=0
        for i,p in enumerate(pairs):
            y=labels.get(i,'s'); if y=='s': continue
            A=torch.from_numpy(load_seg(p['a'])); B=torch.from_numpy(load_seg(p['b']))
            ra=model(A).sum(); rb=model(B).sum()
            loss = -torch.log(torch.sigmoid(ra-rb)+1e-6) if y=='a' else -torch.log(torch.sigmoid(rb-ra)+1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); L+=loss.item(); n+=1
        print(f'epoch {ep+1}: loss={L/max(n,1):.4f}')
    os.makedirs(os.path.dirname(a.out), exist_ok=True); torch.save({'model':model.state_dict(),'din':din}, a.out); print('saved', a.out)
