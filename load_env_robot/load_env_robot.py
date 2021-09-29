import numpy as np
import random
import os
import sys
import signal
import argparse
from argparse import Namespace

from omni.isaac.python_app import OmniKitHelper

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

	from pxr import UsdGeom, Gf, Sdf
	
	from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
	#from omni.isaac.dynamic_control import _dynamic_control
	from omni.physx.scripts import utils
	
	result, nucleus = find_nucleus_server()
	
	if result is False:
		carb.log_error("Could not find nucleus server, exiting")
		exit()
	
	# enable extension
	ext_manager = omni.kit.app.get_app().get_extension_manager()
	ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)
	ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
		
	#load environment
	env_path = nucleus + args.env_path
	print(env_path)
	omni.usd.get_context().open_stage(env_path, None)
	
	#random 10 objects
	stage = kit.get_stage()
	TRANSLATION_RANGE = 1000.0
	SCALE = 30.0
	for i in range(10):
		prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
		prim = stage.DefinePrim(f"/World/cube{i}", prim_type)
		translation = np.random.rand(3) * TRANSLATION_RANGE
		translation[2] = 40.0
		UsdGeom.XformCommonAPI(prim).SetTranslate(translation.tolist())
		UsdGeom.XformCommonAPI(prim).SetScale((SCALE, SCALE, SCALE))
		#prim.GetAttribute("primvars:displayColor").Set([np.random.rand(3).tolist()])
		
		prim_path = stage.GetPrimAtPath(f"/World/cube{i}")
		utils.setRigidBody(prim_path, "convexHull", False)
    
	#load robot
	translation = np.random.rand(3) * TRANSLATION_RANGE
	angle = np.random.rand(1)
	
	prefix = "/World/soap_odom"
	prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
	print(prim_path)
	robot_prim = stage.DefinePrim(prim_path, "Xform")
	robot_prim.GetReferences().AddReference(args.robo_path)
	xform = UsdGeom.Xformable(robot_prim)
	xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
	mat = Gf.Matrix4d().SetTranslate(translation.tolist())
	mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), (angle[0])))
	xform_op.Set(mat)
	
	kit.app.update()
	kit.app.update()

	print("Loading stage...")
	while kit.is_loading():
		kit.update(1.0 / 60.0)
	print("Loading Complete")
	
	omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/odom/robot/agv_lidar/ROS_Lidar.enabled"), value=True, prev=None)
	omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/odom/robot/ROS_PoseTree.enabled"), value=True, prev=None)
	omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/odom/robot/ROS_JointState.enabled"), value=True, prev=None)
	omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/ROS_Clock.enabled"), value=True, prev=None)
	
	kit.play()
	
	while kit.app.is_running():
		# Run in realtime mode, we don't specify the step size
		kit.update()

	kit.stop()
	kit.shutdown()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--headless", help="run in headless mode (no GUI)", action="store_true")
	parser.add_argument("--env_path", type=str, help="Path to environment usd file", required=True)
	parser.add_argument("--robo_path", type=str, help="Path to robot usd file", required=True)
	
	args = parser.parse_args()
	
	print("running with args: ", args)
	
	def handle_exit(*args, **kwargs):
		print("Exiting...")
		quit()

	signal.signal(signal.SIGINT, handle_exit)
	
	Run(args)
