
import os
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema, UsdShade
HERE=os.path.dirname(__file__); OUT=os.path.join(HERE,'assets','dexbench.usd')
def make_material(stage, path, color=(0.8,0.8,0.8), friction=1.0, restitution=0.0):
    mat=UsdShade.Material.Define(stage, path); pbr=UsdShade.Shader.Define(stage, path+'/PBR'); pbr.CreateIdAttr('UsdPreviewSurface')
    pbr.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color)); pbr.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.6); pbr.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(pbr, 'surface'); phys=UsdPhysics.MaterialAPI.Apply(mat.GetPrim()); phys.CreateStaticFrictionAttr().Set(friction); phys.CreateDynamicFrictionAttr().Set(friction); phys.CreateRestitutionAttr().Set(restitution); return mat
def add_finger(stage, root_path, pos, rot_deg, color):
    root=UsdGeom.Xform.Define(stage, root_path); UsdGeom.XformCommonAPI(root).SetTranslate(pos); UsdGeom.XformCommonAPI(root).SetRotate(rot_deg, UsdGeom.XformCommonAPI.RotationOrderXYZ)
    l1=UsdGeom.Capsule.Define(stage, root_path+'/link1'); l1.CreateHeightAttr().Set(0.05); l1.CreateRadiusAttr().Set(0.006); l1.AddTranslateOp().Set(Gf.Vec3f(0,0,0.025)); UsdShade.MaterialBindingAPI(l1).Bind(make_material(stage, root_path+'/m1', color))
    l2=UsdGeom.Capsule.Define(stage, root_path+'/link2'); l2.CreateHeightAttr().Set(0.04); l2.CreateRadiusAttr().Set(0.006); l2.AddTranslateOp().Set(Gf.Vec3f(0,0,0.065)); UsdShade.MaterialBindingAPI(l2).Bind(make_material(stage, root_path+'/m2', color))
    pad=UsdGeom.Sphere.Define(stage, root_path+'/pad'); pad.CreateRadiusAttr().Set(0.008); pad.AddTranslateOp().Set(Gf.Vec3f(0,0,0.085)); UsdShade.MaterialBindingAPI(pad).Bind(make_material(stage, root_path+'/padm', (0.3,0.7,1.0), friction=1.2))
    for p in (l1,l2,pad): UsdPhysics.CollisionAPI.Apply(p.GetPrim()); UsdPhysics.RigidBodyAPI.Apply(p.GetPrim())
    j1=UsdPhysics.RevoluteJoint.Define(stage, root_path+'/hinge1'); j1.CreateAxisAttr('Y'); j1.CreateBody0Rel().SetTargets(['/World']); j1.CreateBody1Rel().SetTargets([l1.GetPath()]); j1.CreateLowerLimitAttr().Set(-1.05); j1.CreateUpperLimitAttr().Set(1.05)
    j2=UsdPhysics.RevoluteJoint.Define(stage, root_path+'/hinge2'); j2.CreateAxisAttr('Y'); j2.CreateBody0Rel().SetTargets([l1.GetPath()]); j2.CreateBody1Rel().SetTargets([l2.GetPath()]); j2.CreateLowerLimitAttr().Set(-1.05); j2.CreateUpperLimitAttr().Set(1.05)
    for j in (j1,j2): d=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular'); d.CreateStiffnessAttr().Set(100.0); d.CreateDampingAttr().Set(5.0); d.CreateTargetPositionAttr().Set(0.0)
def main():
    stage=Usd.Stage.CreateNew(OUT); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdPhysics.SetStageMetersPerUnit(stage,1.0)
    scene=UsdPhysics.Scene.Define(stage, Sdf.Path('/World/Physics')); physx=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()); physx.CreateEnableCCDAttr().Set(True)
    table=UsdGeom.Cube.Define(stage,'/World/Table'); table.CreateSizeAttr().Set(0.02); table.AddScaleOp().Set(Gf.Vec3f(40,30,1)); table.AddTranslateOp().Set(Gf.Vec3f(0,0,0.4)); UsdShade.MaterialBindingAPI(table).Bind(make_material(stage,'/World/MatTable',(0.75,0.75,0.75))); UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    cube=UsdGeom.Cube.Define(stage,'/World/Cube'); cube.CreateSizeAttr().Set(0.04); cube.AddTranslateOp().Set(Gf.Vec3f(0,0,0.45)); UsdShade.MaterialBindingAPI(cube).Bind(make_material(stage,'/World/MatCube',(0.8,0.2,0.2))); UsdPhysics.CollisionAPI.Apply(cube.GetPrim()); UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    add_finger(stage,'/World/FingerA', Gf.Vec3f(0.09,0.0,0.45), Gf.Vec3f(0,0,180),(0.2,0.2,0.8))
    add_finger(stage,'/World/FingerB', Gf.Vec3f(-0.045,0.078,0.45), Gf.Vec3f(0,0,-60),(0.2,0.6,0.3))
    add_finger(stage,'/World/FingerC', Gf.Vec3f(-0.045,-0.078,0.45), Gf.Vec3f(0,0,60),(0.9,0.5,0.2))
    stage.Save(); print('Saved', OUT)
if __name__=='__main__': main()
