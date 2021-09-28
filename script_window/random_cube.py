from omni.physx.scripts import utils

stage = omni.usd.get_context().get_stage()
prim = stage.DefinePrim(f"/World/cube1", "Cube")

UsdGeom.XformCommonAPI(prim).SetTranslate([500.0, 2.0, 60.0])
UsdGeom.XformCommonAPI(prim).SetScale((50.0, 50.0, 50.0))

prim_path = stage.GetPrimAtPath(f"/World/cube1")
utils.setRigidBody(prim_path, "convexHull", False)
