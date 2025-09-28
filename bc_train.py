
import argparse, glob, numpy as np, torch, torch.nn as nn, torch.optim as optim, os
class MLP(nn.Module):
    def __init__(self,din,dout): super().__init__(); self.net=nn.Sequential(nn.Linear(din,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,dout))
    def forward(self,x): return self.net(x)
def load(globpat):
    X=[];Y=[]
    for f in glob.glob(globpat): d=np.load(f,allow_pickle=True); X.append(d['observations']); Y.append(d['actions'])
    return np.concatenate(X,0).astype(np.float32), np.concatenate(Y,0).astype(np.float32)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--demo-glob',required=True); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--bs',type=int,default=1024); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--save',default='runs/bc.pt'); a=ap.parse_args()
    X,Y=load(a.demo_glob); din,dout=X.shape[1],Y.shape[1]; mdl=MLP(din,dout); opt=optim.AdamW(mdl.parameters(),lr=a.lr); lossfn=nn.MSELoss()
    for ep in range(a.epochs):
        idx=np.random.permutation(len(X))
        for i in range(0,len(X),a.bs):
            j=idx[i:i+a.bs]; xb=torch.from_numpy(X[j]); yb=torch.from_numpy(Y[j]); pred=mdl(xb); loss=lossfn(pred,yb); opt.zero_grad(); loss.backward(); opt.step()
        print(f'epoch {ep+1}/{a.epochs} loss={loss.item():.4f}')
    os.makedirs(os.path.dirname(a.save), exist_ok=True); torch.save({'model':mdl.state_dict(),'din':din,'dout':dout}, a.save); print('saved', a.save)
