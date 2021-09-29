# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Generate offline synthetic dataset using two cameras
"""


import asyncio
import copy
import numpy as np
import os
import random
import torch
import signal

import carb
import omni
from omni.isaac.python_app import OmniKitHelper

# Default rendering parameters
RENDER_CONFIG = {
    "renderer": "RayTracedLighting",
    "samples_per_pixel_per_frame": 12,
    "headless": False,
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
}


class RandomScenario(torch.utils.data.IterableDataset):
    def __init__(self, scenario_path, max_queue_size):

        self.kit = OmniKitHelper(config=RENDER_CONFIG)
        from omni.isaac.synthetic_utils import SyntheticDataHelper, DataWriter, DomainRandomization

        self.sd_helper = SyntheticDataHelper()
        self.dr_helper = DomainRandomization()
        self.writer_helper = DataWriter
        self.dr_helper.toggle_manual_mode()
        self.stage = self.kit.get_stage()
        self.result = True

        from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

        if scenario_path is None:
            self.result, nucleus_server = find_nucleus_server()
            if self.result is False:
                carb.log_error("Could not find nucleus server with /Isaac folder")
                return
            self.asset_path = nucleus_server + "/Isaac"
            scenario_path = self.asset_path + "/Samples/Synthetic_Data/Stage/warehouse_with_sensors.usd"
        self.scenario_path = scenario_path
        self.max_queue_size = max_queue_size
        self.data_writer = None

        self._setup_world(scenario_path)
        self.cur_idx = 0
        self.exiting = False
        self._viewport = omni.kit.viewport.get_viewport_interface()
        self._sensor_settings = {}

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        self.exiting = True

    def add_stereo_setup(self):
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        # Create two camera
        center_point = Gf.Vec3d(0, 0, 200)
        stereoPrimPath = "/World/Stereo"
        leftCameraPrimPath = stereoPrimPath + "/LeftCamera"
        rightCameraPrimPath = stereoPrimPath + "/RightCamera"
        self.stereoPrim = stage.DefinePrim(stereoPrimPath, "Xform")
        UsdGeom.XformCommonAPI(self.stereoPrim).SetTranslate(center_point)
        leftCameraPrim = stage.DefinePrim(leftCameraPrimPath, "Camera")
        UsdGeom.XformCommonAPI(leftCameraPrim).SetTranslate(Gf.Vec3d(0, -10, 0))
        UsdGeom.XformCommonAPI(leftCameraPrim).SetRotate(Gf.Vec3f(90, 0, 90))
        rightCameraPrim = stage.DefinePrim(rightCameraPrimPath, "Camera")
        UsdGeom.XformCommonAPI(rightCameraPrim).SetTranslate(Gf.Vec3d(0, 10, 0))
        UsdGeom.XformCommonAPI(rightCameraPrim).SetRotate(Gf.Vec3f(90, 0, 90))

        # Need to set this before setting viewport window size
        carb.settings.acquire_settings_interface().set_int("/app/renderer/resolution/width", -1)
        carb.settings.acquire_settings_interface().set_int("/app/renderer/resolution/height", -1)
        # Get existing viewport, set active camera as left camera
        viewport_handle_1 = omni.kit.viewport.get_viewport_interface().get_instance("Viewport")
        viewport_window_1 = omni.kit.viewport.get_viewport_interface().get_viewport_window(viewport_handle_1)
        viewport_window_1.set_texture_resolution(1280, 720)
        viewport_window_1.set_active_camera(leftCameraPrimPath)
        # Create new viewport, set active camera as right camera
        viewport_handle_2 = omni.kit.viewport.get_viewport_interface().create_instance()
        viewport_window_2 = omni.kit.viewport.get_viewport_interface().get_viewport_window(viewport_handle_2)
        viewport_window_2.set_active_camera("/World/Stereo/RightCamera")
        viewport_window_2.set_texture_resolution(1280, 720)
        viewport_window_2.set_window_pos(720, 0)
        viewport_window_2.set_window_size(720, 890)

        # Setup stereo camera movement randomization
        radius = 100
        target_points_list = []
        for theta in range(200, 300):
            th = theta * np.pi / 180
            x = radius * np.cos(th) + center_point[0]
            y = radius * np.sin(th) + center_point[1]
            target_points_list.append(Gf.Vec3f(x, y, center_point[2]))
        lookat_target_points_list = [a for a in target_points_list[1:]]
        lookat_target_points_list.append(target_points_list[0])
        result, prim = omni.kit.commands.execute(
            "CreateTransformComponentCommand",
            prim_paths=[stereoPrimPath],
            target_points=target_points_list,
            lookat_target_points=lookat_target_points_list,
            enable_sequential_behavior=True,
        )

    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)

    def _setup_world(self, scenario_path):
        # Load scenario
        setup_task = asyncio.ensure_future(self.load_stage(scenario_path))
        while not setup_task.done():
            self.kit.update()
        self.add_stereo_setup()
        self.kit.update()
        self.kit.setup_renderer()
        self.kit.update()

    def __iter__(self):
        return self

    def __next__(self):
        # step once and then wait for materials to load
        self.dr_helper.randomize_once()
        self.kit.update()
        while self.kit.is_loading():
            self.kit.update()

        # Enable/disable sensor output and their format
        sensor_settings_viewport_1 = {
            "rgb": {"enabled": True},
            "depth": {"enabled": True, "colorize": True, "npy": True},
            "instance": {"enabled": True, "colorize": True, "npy": True},
            "semantic": {"enabled": True, "colorize": True, "npy": True},
            "bbox_2d_tight": {"enabled": True, "colorize": True, "npy": True},
            "bbox_2d_loose": {"enabled": True, "colorize": True, "npy": True},
        }
        sensor_settings_viewport_2 = {
            "rgb": {"enabled": True},
            "depth": {"enabled": True, "colorize": True, "npy": True},
            "instance": {"enabled": True, "colorize": True, "npy": True},
            "semantic": {"enabled": True, "colorize": True, "npy": True},
            "bbox_2d_tight": {"enabled": True, "colorize": True, "npy": True},
            "bbox_2d_loose": {"enabled": True, "colorize": True, "npy": True},
        }
        viewports = self._viewport.get_instance_list()
        self._viewport_names = [self._viewport.get_viewport_window_name(vp) for vp in viewports]
        # Make sure two viewports are initialized
        if len(self._viewport_names) != 2:
            return
        self._sensor_settings[self._viewport_names[0]] = copy.deepcopy(sensor_settings_viewport_1)
        self._sensor_settings[self._viewport_names[1]] = copy.deepcopy(sensor_settings_viewport_2)
        self._num_worker_threads = 4
        self._output_folder = os.getcwd() + "/output"

        # Write to disk
        if self.data_writer is None:
            self.data_writer = self.writer_helper(
                self._output_folder, self._num_worker_threads, self.max_queue_size, self._sensor_settings
            )
            self.data_writer.start_threads()

        image = None
        for viewport_name in self._viewport_names:
            groundtruth = {
                "METADATA": {
                    "image_id": str(self.cur_idx),
                    "viewport_name": viewport_name,
                    "DEPTH": {},
                    "INSTANCE": {},
                    "SEMANTIC": {},
                    "BBOX2DTIGHT": {},
                    "BBOX2DLOOSE": {},
                },
                "DATA": {},
            }

            gt_list = []
            if self._sensor_settings[viewport_name]["rgb"]["enabled"]:
                gt_list.append("rgb")
            if self._sensor_settings[viewport_name]["depth"]["enabled"]:
                gt_list.append("depthLinear")
            if self._sensor_settings[viewport_name]["bbox_2d_tight"]["enabled"]:
                gt_list.append("boundingBox2DTight")
            if self._sensor_settings[viewport_name]["bbox_2d_loose"]["enabled"]:
                gt_list.append("boundingBox2DLoose")
            if self._sensor_settings[viewport_name]["instance"]["enabled"]:
                gt_list.append("instanceSegmentation")
            if self._sensor_settings[viewport_name]["semantic"]["enabled"]:
                gt_list.append("semanticSegmentation")

            # Render new frame
            self.kit.update()

            # Collect Groundtruth
            viewport = self._viewport.get_viewport_window(self._viewport.get_instance(viewport_name))
            gt = self.sd_helper.get_groundtruth(gt_list, viewport)

            # RGB
            image = gt["rgb"]
            if self._sensor_settings[viewport_name]["rgb"]["enabled"] and gt["state"]["rgb"]:
                groundtruth["DATA"]["RGB"] = gt["rgb"]

            # Depth
            if self._sensor_settings[viewport_name]["depth"]["enabled"] and gt["state"]["depthLinear"]:
                groundtruth["DATA"]["DEPTH"] = gt["depthLinear"].squeeze()
                groundtruth["METADATA"]["DEPTH"]["COLORIZE"] = self._sensor_settings[viewport_name]["depth"]["colorize"]
                groundtruth["METADATA"]["DEPTH"]["NPY"] = self._sensor_settings[viewport_name]["depth"]["npy"]

            # Instance Segmentation
            if self._sensor_settings[viewport_name]["instance"]["enabled"] and gt["state"]["instanceSegmentation"]:
                instance_data = gt["instanceSegmentation"][0]
                groundtruth["DATA"]["INSTANCE"] = instance_data
                groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = instance_data.shape[1]
                groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = instance_data.shape[0]
                groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = self._sensor_settings[viewport_name]["instance"][
                    "colorize"
                ]
                groundtruth["METADATA"]["INSTANCE"]["NPY"] = self._sensor_settings[viewport_name]["instance"]["npy"]

            # Semantic Segmentation
            if self._sensor_settings[viewport_name]["semantic"]["enabled"] and gt["state"]["semanticSegmentation"]:
                semantic_data = gt["semanticSegmentation"]
                semantic_data[semantic_data == 65535] = 0  # deals with invalid semantic id
                groundtruth["DATA"]["SEMANTIC"] = semantic_data
                groundtruth["METADATA"]["SEMANTIC"]["WIDTH"] = semantic_data.shape[1]
                groundtruth["METADATA"]["SEMANTIC"]["HEIGHT"] = semantic_data.shape[0]
                groundtruth["METADATA"]["SEMANTIC"]["COLORIZE"] = self._sensor_settings[viewport_name]["semantic"][
                    "colorize"
                ]
                groundtruth["METADATA"]["SEMANTIC"]["NPY"] = self._sensor_settings[viewport_name]["semantic"]["npy"]

            # 2D Tight BBox
            if self._sensor_settings[viewport_name]["bbox_2d_tight"]["enabled"] and gt["state"]["boundingBox2DTight"]:
                groundtruth["DATA"]["BBOX2DTIGHT"] = gt["boundingBox2DTight"]
                groundtruth["METADATA"]["BBOX2DTIGHT"]["COLORIZE"] = self._sensor_settings[viewport_name][
                    "bbox_2d_tight"
                ]["colorize"]
                groundtruth["METADATA"]["BBOX2DTIGHT"]["NPY"] = self._sensor_settings[viewport_name]["bbox_2d_tight"][
                    "npy"
                ]

            # 2D Loose BBox
            if self._sensor_settings[viewport_name]["bbox_2d_loose"]["enabled"] and gt["state"]["boundingBox2DLoose"]:
                groundtruth["DATA"]["BBOX2DLOOSE"] = gt["boundingBox2DLoose"]
                groundtruth["METADATA"]["BBOX2DLOOSE"]["COLORIZE"] = self._sensor_settings[viewport_name][
                    "bbox_2d_loose"
                ]["colorize"]
                groundtruth["METADATA"]["BBOX2DLOOSE"]["NPY"] = self._sensor_settings[viewport_name]["bbox_2d_loose"][
                    "npy"
                ]

            self.data_writer.q.put(groundtruth)

        self.cur_idx += 1
        return image


if __name__ == "__main__":
    "Typical usage"
    import argparse

    parser = argparse.ArgumentParser("Stereo dataset generator")
    parser.add_argument("--scenario", type=str, help="Scenario to load from omniverse server")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to record")
    parser.add_argument("--max_queue_size", type=int, default=500, help="Max size of queue to store and process data")
    args = parser.parse_args()

    dataset = RandomScenario(args.scenario, args.max_queue_size)

    if dataset.result:
        # Iterate through dataset and visualize the output
        print("Loading materials. Will generate data soon...")
        for image in dataset:
            print("ID: ", dataset.cur_idx)
            if dataset.cur_idx == args.num_frames:
                break
            if dataset.exiting:
                break
        # cleanup
        dataset.kit.shutdown()
