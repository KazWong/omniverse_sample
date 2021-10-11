from omni.physx.scripts import utils

stage = omni.usd.get_context().get_stage()
omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
cube_prim = stage.GetPrimAtPath("/World/Cube")
UsdGeom.XformCommonAPI(cube_prim).SetTranslate((0,0,100.0))
utils.setRigidBody(cube_prim, "convexHull", False)

    
import carb
import omni.physx
from omni.physx import get_physx_scene_query_interface

counter = 0

# Defines a cubic region to check overlap with
extent = carb.Float3(200.0, 200.0, 200.0)
origin = carb.Float3(0.0, 0.0, 0.0)
rotation = carb.Float4(0.0, 0.0, 1.0, 0.0)

def report_hit(hit):
    return True
    
while True:
# physX query to detect number of hits for a cubic region
    numHits = get_physx_scene_query_interface().overlap_box(extent, origin, rotation, report_hit, False)
    print(numHits)
# physX query to detect number of hits for a spherical region
# numHits = get_physx_scene_query_interface().overlap_sphere(radius, origin, self.report_hit, False)
    if numHits > 0:
        print("collide")
        break
