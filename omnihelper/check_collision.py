import numpy as np
import random
import os
import sys
import signal
import argparse
from argparse import Namespace

from omni.isaac.python_app import OmniKitHelper

def check_overlap_box():
    # Defines a cubic region to check overlap with
    import omni.physx
    from omni.physx import get_physx_scene_query_interface
    import carb

    #print("*"*50)        
    extent = carb.Float3(50.0, 50.0, 50.0)
    origin = carb.Float3(0.0, 0.0, 0.0)
    rotation = carb.Float4(0.0, 0.0, 1.0, 0.0)
    # physX query to detect number of hits for a cubic region
    numHits = get_physx_scene_query_interface().overlap_box(extent, origin, rotation, report_hit, False)
    print("num of overlaps ", numHits)
    # physX query to detect number of hits for a spherical region
    # numHits = get_physx_scene_query_interface().overlap_sphere(radius, origin, self.report_hit, False)
    #self.kit.update()
    return numHits > 1

def report_hit(hit):
    #from pxr import UsdGeom, Gf, Vt
    #stage = kit.get_stage()
    ## When a collision is detected, the object colour changes to red.
    #hitColor = Vt.Vec3fArray([Gf.Vec3f(180.0 / 255.0, 16.0 / 255.0, 0.0)])
    #usdGeom = UsdGeom.Mesh.Get(stage, hit.rigid_body)
    #usdGeom.GetDisplayColorAttr().Set(hitColor)
    return True
    
def Run(args):
	startup_config = {
		"renderer": "RayTracedLighting",
		"headless": args.headless,
		"experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
	}
	kit = OmniKitHelper(startup_config)
	
	#include after kit
	import carb
	import omni
	import omni.kit.app

	from pxr import UsdGeom, Gf, Sdf, UsdPhysics
	
	from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
	from omni.physx.scripts import utils
	result, nucleus = find_nucleus_server()
	
	if result is False:
		carb.log_error("Could not find nucleus server, exiting")
		exit()
		
	ext_manager = omni.kit.app.get_app().get_extension_manager()
	ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)
	ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
	
	env_path = nucleus + args.env_path
	print(env_path)
	omni.usd.get_context().open_stage(env_path, None)
	stage = kit.get_stage()
	omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
	cube_prim = stage.GetPrimAtPath("/World/Cube")
	UsdGeom.XformCommonAPI(cube_prim).SetTranslate((0,0,100))
	utils.setRigidBody(cube_prim, "convexHull", False)
		
	kit.app.update()
	kit.app.update()

	print("Loading stage...")
	while kit.is_loading():
		kit.update(1.0 / 60.0)
	print("Loading Complete")
	
	kit.play()

	while kit.app.is_running():
		# Run in realtime mode, we don't specify the step size
		if check_overlap_box() == True:
		#    # reset robot to origin
		    print("colide!!")
		kit.update()

	kit.stop()
	kit.shutdown()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--headless", help="run in headless mode (no GUI)", action="store_true")
	parser.add_argument("--env_path", type=str, help="Path to environment usd file", required=True)
	args = parser.parse_args()
	Run(args)
