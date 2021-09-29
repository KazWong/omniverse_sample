# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.python_app import OmniKitHelper
import carb
import omni

# This sample loads a usd stage and creates a robot engine bridge application and starts simulation
# Disposes average fps of the simulation for given time
# Useful for testing an Isaac SDK sample scene using python
CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": False,
    "renderer": "RayTracedLighting",
}


class UsdLoadSample:
    def __init__(self, args):
        CONFIG["headless"] = args.headless
        self.kit = OmniKitHelper(config=CONFIG)
        self.usd_path = ""
        self._viewport = omni.kit.viewport.get_viewport_interface()

    def start(self):
        self.kit.play()

    def stop(self):
        self.kit.stop()
        omni.kit.commands.execute("RobotEngineBridgeDestroyApplication")
        self.kit.shutdown()

    def load_stage(self, args):
        from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return False
        self._asset_path = nucleus_server + "/Isaac"
        self.usd_path = self._asset_path + args.usd_path
        omni.usd.get_context().open_stage(self.usd_path, None)
        # Wait two frames so that stage starts loading
        self.kit.app.update()
        self.kit.app.update()
        return True

    def configure_bridge(self, json_file: str = "isaacsim.app.json"):
        """
        Configure the SDK bridge application that publishes data over tcp
        """
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("omni.isaac.robot_engine_bridge")
        reb_extension_path = ext_manager.get_extension_path(ext_id)
        app_file = f"{reb_extension_path}/resources/isaac_engine/json/{json_file}"
        carb.log_info(f"create application with: {reb_extension_path} {app_file}")
        return omni.kit.commands.execute(
            "RobotEngineBridgeCreateApplication", asset_path=reb_extension_path, app_file=app_file
        )

    def disable_existing_reb_cameras(self):
        """
        Disable existing REB_Camera prims for perf testing
        """
        import omni.isaac.RobotEngineBridgeSchema as REBSchema

        stage = self.kit.get_stage()
        for prim in stage.Traverse():
            if prim.IsA(REBSchema.RobotEngineCamera):
                reb_camera_prim = REBSchema.RobotEngineCamera(prim)
                reb_camera_prim.GetEnabledAttr().Set(False)

    def create_reb_camera(self, cameraIndex, name, width, height):
        """Create a new REB camera in the stage"""
        from pxr import Gf

        result, reb_camera_prim = omni.kit.commands.execute(
            "RobotEngineBridgeCreateCamera",
            path="/World/REB_Camera",
            parent=None,
            rgb_output_component="output",
            rgb_output_channel="encoder_color_{}".format(cameraIndex),
            depth_output_component="output",
            depth_output_channel="encoder_depth_{}".format(cameraIndex),
            segmentation_output_component="output",
            segmentation_output_channel="encoder_segmentation_{}".format(cameraIndex),
            bbox2d_output_component="output",
            bbox2d_output_channel="encoder_bbox_{}".format(cameraIndex),
            bbox2d_class_list="",
            bbox3d_output_component="output",
            bbox3d_output_channel="encoder_bbox3d_{}".format(cameraIndex),
            bbox3d_class_list="",
            rgb_enabled=True,
            depth_enabled=False,
            segmentaion_enabled=True,
            bbox2d_enabled=False,
            bbox3d_enabled=False,
            camera_prim_rel=["{}".format(name)],
            resolution=Gf.Vec2i(int(width), int(height)),
        )


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser("Usd Load sample")
    parser.add_argument("--usd_path", type=str, help="Path to usd file", required=True)
    parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
    parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
    parser.add_argument("--benchmark", default=False, action="store_true", help="Run in benchmark mode")
    parser.add_argument(
        "--benchmark_timeout", type=int, default=60, help="Total walltime in seconds to calculate average FPS for"
    )
    parser.add_argument(
        "--add_rebcamera",
        nargs="*",
        type=str,
        default=[],
        help="Total number of REB Camera prims to add, existing ones will be disabled if this option is specified",
    )

    args, unknown = parser.parse_known_args()
    sample = UsdLoadSample(args)
    if sample.load_stage(args):
        print("Loading stage...")
        while sample.kit.is_loading():
            sample.kit.update(1.0 / 60.0)
        print("Loading Complete")
        # Add parameterized rebcamera along with viewport
        if args.add_rebcamera is not None and len(args.add_rebcamera) > 0:
            # disable existing cameras if we are making new ones
            sample.disable_existing_reb_cameras()
            reb_count = 0
            for name in args.add_rebcamera:
                info = name.split(",")
                sample.create_reb_camera(reb_count, info[0], info[1], info[2])
                reb_count = reb_count + 1
        sample.configure_bridge()
        sample.start()
        if args.test is True:
            for i in range(10):
                sample.kit.update()
            sample.stop()
        elif args.benchmark is True:
            # Warm up simulation
            while sample._viewport.get_viewport_window().get_fps() < 1:
                sample.kit.update(1.0 / 60.0)

            fps_count = 0
            start_time = time.perf_counter()
            end_time = start_time + args.benchmark_timeout
            count = 0
            # Calculate average fps
            while sample.kit.app.is_running() and end_time > time.perf_counter():
                sample.kit.update(1.0 / 60.0)
                fps = sample._viewport.get_viewport_window().get_fps()
                fps_count = fps_count + fps
                count = count + 1
            sample.stop()
            print(f"\n----------- Avg. FPS over {args.benchmark_timeout} sec : {fps_count/count}-----------")
        else:
            while sample.kit.app.is_running():
                # Run in realtime mode, we don't specify the step size
                sample.kit.update()
            sample.stop()
