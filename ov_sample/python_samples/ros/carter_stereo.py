# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import carb
from omni.isaac.python_app import OmniKitHelper

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": False,
}

if __name__ == "__main__":
    # Example ROS bridge sample showing manual control over messages
    kit = OmniKitHelper(config=CONFIG)
    import omni
    from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
    from pxr import Sdf

    # enable ROS bridge extension
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)

    # Locate /Isaac folder on nucleus server to load sample

    result, nucleus_server = find_nucleus_server()
    if result is False:
        carb.log_error("Could not find nucleus server with /Isaac folder, exiting")
        exit()
    usd_path = nucleus_server + "/Isaac/Samples/ROS/Scenario/carter_warehouse_navigation.usd"
    omni.usd.get_context().open_stage(usd_path, None)
    # Wait two frames so that stage starts loading
    kit.app.update()
    kit.app.update()

    print("Loading stage...")
    while kit.is_loading():
        kit.update(1.0 / 60.0)
    print("Loading Complete")

    # Disable all ROS components so we can demonstrate publishing manually
    # Otherwise, if a component is enabled, it will publish every timestep
    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=Sdf.Path("/World/Carter_ROS/ROS_Camera_Stereo_Right.enabled"),
        value=False,
        prev=None,
    )
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path("/World/Carter_ROS/ROS_Camera_Stereo_Left.enabled"), value=False, prev=None
    )
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path("/World/Carter_ROS/ROS_Lidar.enabled"), value=False, prev=None
    )
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path("/World/Carter_ROS/ROS_DifferentialBase.enabled"), value=False, prev=None
    )
    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=Sdf.Path("/World/Carter_ROS/ROS_Carter_Lidar_Broadcaster.enabled"),
        value=False,
        prev=None,
    )
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path("/World/Carter_ROS/ROS_Carter_Broadcaster.enabled"), value=False, prev=None
    )
    omni.kit.commands.execute("ChangeProperty", prop_path=Sdf.Path("/World/ROS_Clock.enabled"), value=False, prev=None)
    kit.play()
    kit.update(1.0 / 60.0)
    # Tick all of the components once to make sure all of the ROS nodes are initialized
    # For cameras this also handles viewport initialization etc.
    omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Camera_Stereo_Right")
    omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Camera_Stereo_Left")
    omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Lidar")
    omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_DifferentialBase")
    omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Carter_Lidar_Broadcaster")
    omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Carter_Broadcaster")
    omni.kit.commands.execute("RosBridgeTickComponent", path="/World/ROS_Clock")
    # Simulate for one second to warm up sim and let everything settle
    for frame in range(60):
        kit.update(1.0 / 60.0)

    # Dock the second camera window
    right_viewport = omni.ui.Workspace.get_window("Viewport")
    left_viewport = omni.ui.Workspace.get_window("Viewport_2")
    if right_viewport is not None and left_viewport is not None:
        left_viewport.dock_in(right_viewport, omni.ui.DockPosition.LEFT)

    # Create a rostopic to publish message to spin robot in place
    # Note that this is not the system level rospy, but one compiled for omniverse
    from geometry_msgs.msg import Twist
    import rospy

    rospy.init_node("carter_stereo", anonymous=True, disable_signals=True, log_level=rospy.ERROR)
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)

    frame = 0
    while kit.app.is_running():
        # Run with a fixed step size
        kit.update(1.0 / 60.0)
        # Publish clock every frame
        omni.kit.commands.execute("RosBridgeTickComponent", path="/World/ROS_Clock")
        # publish TF and Lidar every 2 frames
        if frame % 2 == 0:
            omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Lidar")
            omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_DifferentialBase")
            omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Carter_Lidar_Broadcaster")
            omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Carter_Broadcaster")
            # because we only tick the differential base component every two frames, we can also publish the ROS message at the same rate
            message = Twist()
            message.angular.z = 0.2  # spin in place
            pub.publish(message)
        # Publish cameras every 60 frames or one second of simulation
        if frame % 60 == 0:
            omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Camera_Stereo_Right")
            omni.kit.commands.execute("RosBridgeTickComponent", path="/World/Carter_ROS/ROS_Camera_Stereo_Left")

        frame = frame + 1
    pub.unregister()
    rospy.signal_shutdown("carter_stereo complete")
    kit.stop()
    kit.shutdown()
