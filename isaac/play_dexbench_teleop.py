
import argparse, os, numpy as np
from omni.isaac.kit import SimulationApp
parser=argparse.ArgumentParser(); parser.add_argument('--usd',type=str,default=None); parser.add_argument('--episodes',type=int,default=1); parser.add_argument('--len',type=int,default=1200); parser.add_argument('--headless',action='store_true'); args,_=parser.parse_known_args()
simulation_app=SimulationApp({'headless':args.headless})
from omni.isaac.core import World; from omni.isaac.core.utils.stage import open_stage, get_current_stage; from omni.isaac.core.utils.viewports import set_camera_view; from omni.isaac.core.utils.physics import set_physics_dt
from utils_isaac import clamp, smooth, set_drive_target
HERE=os.path.dirname(__file__); usd=args.usd or os.path.join(HERE,'assets','dexbench.usd'); open_stage(usd); stage=get_current_stage(); world=World(stage_units_in_meters=1.0,rendering_dt=1/60.0,physics_dt=1/240.0); set_physics_dt(1/240.0); set_camera_view([0.35,0.45,0.55],[0,0,0.45])
J=['/World/FingerA/hinge1','/World/FingerA/hinge2','/World/FingerB/hinge1','/World/FingerB/hinge2','/World/FingerC/hinge1','/World/FingerC/hinge2']; ACT=6
def lims(jp): j=stage.GetPrimAtPath(jp); return float(j.GetAttribute('physics:lowerLimit').Get()), float(j.GetAttribute('physics:upperLimit').Get())
L=[lims(jp) for jp in J]
def map_a(a): return [0.5*(lo+hi)+0.5*(hi-lo)*float(v) for (lo,hi),v in zip(L,a)]
def input_keyboard():
    import carb.input as ci; dev=ci.acquire_input_device(ci.DeviceType.KEYBOARD,0); p=lambda k:ci.is_key_down(dev,k); a=np.zeros(ACT,np.float32)
    a[0]+=1 if p(ci.KeyboardInput.A) else 0; a[0]-=1 if p(ci.KeyboardInput.Z) else 0
    a[1]+=1 if p(ci.KeyboardInput.S) else 0; a[1]-=1 if p(ci.KeyboardInput.X) else 0
    a[2]+=1 if p(ci.KeyboardInput.K) else 0; a[2]-=1 if p(ci.KeyboardInput.COMMA) else 0
    a[3]+=1 if p(ci.KeyboardInput.L) else 0; a[3]-=1 if p(ci.KeyboardInput.PERIOD) else 0
    a[4]+=1 if p(ci.KeyboardInput.V) else 0; a[4]-=1 if p(ci.KeyboardInput.B) else 0
    a[5]+=1 if p(ci.KeyboardInput.N) else 0; a[5]-=1 if p(ci.KeyboardInput.M) else 0
    return a, p(ci.KeyboardInput.SPACE), p(ci.KeyboardInput.ESCAPE) or p(ci.KeyboardInput.Q)
def main():
    a=np.zeros(ACT,np.float32)
    for ep in range(args.episodes):
        world.reset(); t=0; run=True
        while run and t<args.len:
            world.step(render=True)
            v,save,quitk = input_keyboard(); a=smooth(a,v,0.85); a=clamp(a,-1,1)
            for jp,tar in zip(J,map_a(a)): set_drive_target(stage,jp,tar)
            t+=1
            if save or t>=args.len: run=False
            if quitk: break
if __name__=='__main__': main(); simulation_app.close()
