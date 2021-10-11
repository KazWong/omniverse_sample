import numpy as np
import random
import os
import sys
import signal
import argparse
from argparse import Namespace
from test_env import Env_config
from test_robot import Robot_config

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

	from pxr import UsdGeom, Gf, Sdf, UsdPhysics
	
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
	test_env = Env_config(omni,kit)
	# create objects
	obj_list = test_env.create_objects(4,4,4)
	import omni.isaac.dr as dr
	dr_interface = dr._dr.acquire_dr_interface()
	#print(obj_list)
	# domain randomization
	test_env.domain_randomization_test(obj_list)
	# load robot
	stage = kit.get_stage()
	TRANSLATION_RANGE = 1000.0
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
	DRIVE_STIFFNESS = 10000.0
	# Set joint drive parameters
	wheel_back_left_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_back_left_joint"), "angular")
	wheel_back_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
	
	wheel_back_right_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_back_right_joint"), "angular")
	wheel_back_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
	
	wheel_front_left_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_front_left_joint"), "angular")
	wheel_front_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
	
	wheel_front_right_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_front_right_joint"), "angular")
	wheel_front_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
	kit.app.update()
	kit.app.update()

	print("Loading stage...")
	while kit.is_loading():
		kit.update(1.0 / 60.0)
	print("Loading Complete")
	
	#from omni.isaac import RosBridgeSchema
	#omni.kit.commands.execute('ROSBridgeCreatePoseTree', path='/World/soap_odom/ROS_PoseTree', parent=None)
	#omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/ROS_PoseTree.enabled"), value=True, prev=None)
	#omni.kit.commands.execute('RosBridgeCreatePrim', path='/ROS_PoseTree', parent=None, enabled=True, scehma_type=<class 'omni.isaac.RosBridgeSchema.RosPoseTree'>)
	# add ros joint state
	#omni.kit.commands.execute('ROSBridgeCreateJointState', path='/World/soap_odom/ROS_JointState', parent=None)
	#omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/ROS_JointState.enabled"), value=True, prev=None)
    #omni.kit.commands.execute('RosBridgeCreatePrim', path='/World/soap_odom/ROS_JointState', parent=None, enabled=True, scehma_type=<class 'omni.isaac.RosBridgeSchema.RosJointState'>)

	# add ros lidar
	#omni.kit.commands.execute('ROSBridgeCreateLidar', path='/World/soap_odom/agv_lidar/ROS_Lidar', parent=None)
	#omni.kit.commands.execute('RosBridgeCreatePrim', path='/World/soap_odom/agv_lidar/ROS_Lidar', parent=None, enabled=True, scehma_type=<class 'omni.isaac.RosBridgeSchema.RosLidar'>)
	
	# add ros clock
	omni.kit.commands.execute('ROSBridgeCreateClock',path='/ROS_Clock',parent=None)
	omni.kit.commands.execute('ChangeProperty', prop_path=Sdf.Path('/World/ROS_Clock.queueSize'), value=0, prev=10)
	
	#omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/agv_lidar/ROS_Lidar.enabled"), value=True, prev=None)
	#omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/ROS_PoseTree.enabled"), value=True, prev=None)
	#omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/soap_odom/ROS_JointState.enabled"), value=True, prev=None)
	#omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/ROS_Clock.enabled"), value=True, prev=None)
	
	kit.play()
	test_rob = Robot_config(stage, omni, robot_prim)
	# initial robot
	test_rob.teleport((0,0,30), 0)
	while kit.app.is_running():
		# Run in realtime mode, we don't specify the step size
		if test_rob.check_overlap_box() == True:
		#    # reset robot to origin
		    print("colide!!, reset robot")
		    test_rob.teleport((0,0,30), 0)
		    dr_interface.randomize_once()
		#test_rob.check_overlap_box()
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
