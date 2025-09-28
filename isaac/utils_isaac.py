
import numpy as np
from pxr import UsdPhysics
def clamp(v, lo, hi): return np.minimum(np.maximum(v, lo), hi)
def smooth(prev, target, k=0.85): return k*prev + (1-k)*target
def set_drive_target(stage, joint_path, target):
    jprim = stage.GetPrimAtPath(joint_path)
    try:
        drv = UsdPhysics.DriveAPI(jprim, "angular")
        drv.CreateTargetPositionAttr().Set(float(target))
    except Exception as e:
        print("Drive target error:", e)
