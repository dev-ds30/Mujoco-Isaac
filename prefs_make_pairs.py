
import argparse, glob, json, numpy as np, os
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--demo-glob',required=True); ap.add_argument('--seg-len',type=int,default=150); ap.add_argument('--pairs',type=int,default=40); ap.add_argument('--out',default='demos/preferences/pairs.json'); a=ap.parse_args()
    demos=sorted(glob.glob(a.demo_glob)); rng=np.random.default_rng(0); pairs=[]
    for _ in range(a.pairs):
        f=rng.choice(demos); T=len(np.load(f,allow_pickle=True)['actions']); s1=int(rng.integers(0, max(1,T-a.seg_len))); s2=int(rng.integers(0, max(1,T-a.seg_len)))
        pairs.append({"a":{"file":f,"start":s1,"end":s1+a.seg_len},"b":{"file":f,"start":s2,"end":s2+a.seg_len}})
    os.makedirs(os.path.dirname(a.out),exist_ok=True); json.dump({"pairs":pairs}, open(a.out,"w"), indent=2); print("wrote", a.out)
