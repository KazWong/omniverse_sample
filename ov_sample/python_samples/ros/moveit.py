# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import carb
from omni.isaac.python_app import OmniKitHelper

FRANKA_STAGE_PATH = "/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": False,
}


def wait_load_stage():
    # Wait two frames so stage starts loading
    kit.app.update()
    kit.app.update()

    print("Loading stage...")
    while kit.is_loading():
        kit.update(1.0 / 60.0)
    print("Loading Complete")


if __name__ == "__main__":
    # Example ROS bridge sample demonstrating the manual loading of stages
    # and creation of ROS components
    kit = OmniKitHelper(config=CONFIG)
    import omni
    from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
    from omni.isaac.utils.scripts.scene_utils import create_background
    from pxr import Gf

    # enable ROS bridge extension
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)

    # Locate /Isaac folder on nucleus server to load environment and robot stages
    result, _nucleus_path = find_nucleus_server()
    if result is False:
        carb.log_error("Could not find nucleus server with /Isaac folder, exiting")
        exit()

    # Initialize extension and UI elements
    _viewport = omni.kit.viewport.get_default_viewport_window()
    _usd_context = omni.usd.get_context()

    # Preparing stage
    _viewport.set_camera_position("/OmniverseKit_Persp", 120, 120, 80, True)
    _viewport.set_camera_target("/OmniverseKit_Persp", 0, 0, 50, True)
    _stage = _usd_context.get_stage()

    # Loading the simple_room environment
    background_asset_path = _nucleus_path + BACKGROUND_USD_PATH
    create_background(_stage, background_asset_path, background_path=BACKGROUND_STAGE_PATH, offset=Gf.Vec3d(0, 0, 0))

    wait_load_stage()

    # Loading the franka robot USD
    franka_asset_path = _nucleus_path + FRANKA_USD_PATH
    prim = _stage.DefinePrim(FRANKA_STAGE_PATH, "Xform")
    prim.GetReferences().AddReference(franka_asset_path)
    rot_mat = Gf.Matrix3d(Gf.Rotation((0, 0, 1), 90))
    omni.kit.commands.execute(
        "TransformPrimCommand",
        path=prim.GetPath(),
        old_transform_matrix=None,
        new_transform_matrix=Gf.Matrix4d().SetRotate(rot_mat).SetTranslateOnly(Gf.Vec3d(0, -64, 0)),
    )

    wait_load_stage()

    # Loading all ROS components initially as disabled so we can demonstrate publishing manually
    # Otherwise, if a component is enabled, it will publish every timestep

    # Load ROS Clock
    omni.kit.commands.execute("ROSBridgeCreateClock", path="/ROS_Clock", enabled=False)

    # Load Joint State
    omni.kit.commands.execute(
        "ROSBridgeCreateJointState", path="/ROS_JointState", articulation_prim_rel=[FRANKA_STAGE_PATH], enabled=False
    )

    # Load Pose Tree
    omni.kit.commands.execute(
        "ROSBridgeCreatePoseTree", path="/ROS_PoseTree", target_prims_rel=[FRANKA_STAGE_PATH], enabled=False
    )

    kit.play()
    kit.update(1.0 / 60.0)

    # Tick all of the components once to make sure all of the ROS nodes are initialized
    omni.kit.commands.execute("RosBridgeTickComponent", path="/ROS_JointState")
    omni.kit.commands.execute("RosBridgeTickComponent", path="/ROS_PoseTree")
    omni.kit.commands.execute("RosBridgeTickComponent", path="/ROS_Clock")

    # Simulate for one second to warm up sim and let everything settle
    for frame in range(60):
        kit.update(1.0 / 60.0)

    kit.play()
    while kit.app.is_running():

        # Run with a fixed step size
        kit.update(1.0 / 60.0)

        # Publish clock, TF and JointState each frame
        omni.kit.commands.execute("RosBridgeTickComponent", path="/ROS_Clock")
        omni.kit.commands.execute("RosBridgeTickComponent", path="/ROS_JointState")
        omni.kit.commands.execute("RosBridgeTickComponent", path="/ROS_PoseTree")

    kit.stop()
    kit.shutdown()
