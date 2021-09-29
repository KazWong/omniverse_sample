# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import random
import os
import omni
from omni.isaac.python_app import OmniKitHelper
import carb.tokens
import argparse

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": True,
    "renderer": "RayTracedLighting",
}

# D435
FOCAL_LEN = 1.93
HORIZONTAL_APERTURE = 2.682
VERTICAL_APERTURE = 1.509
FOCUS_DIST = 400

RANDOMIZE_SCENE_EVERY_N_STEPS = 10


class DualCameraSample:
    def __init__(self):
        self.kit = OmniKitHelper(config=CONFIG)
        import omni.physx
        from pxr import UsdGeom, Usd, Gf
        from omni.isaac.synthetic_utils import DomainRandomization
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from omni.isaac.robot_engine_bridge import _robot_engine_bridge

        self._re_bridge = _robot_engine_bridge.acquire_robot_engine_bridge_interface()
        self._viewport = omni.kit.viewport.get_viewport_interface()

        self.dr_helper = DomainRandomization()
        self.sd_helper = SyntheticDataHelper()
        self.frame = 0
        self.Gf = Gf
        self.UsdGeom = UsdGeom
        self.Usd = Usd

    def shutdown(self):
        self.kit.shutdown()

    def start(self):
        self.kit.play()

    def stop(self):
        self.kit.stop()
        omni.kit.commands.execute("RobotEngineBridgeDestroyApplication")

    def create_stage(self):
        # open base stage and set up axis to Z
        stage = self.kit.get_stage()
        rootLayer = stage.GetRootLayer()
        rootLayer.SetPermissionToEdit(True)
        with self.Usd.EditContext(stage, rootLayer):
            self.UsdGeom.SetStageUpAxis(stage, self.UsdGeom.Tokens.z)
        # make two prims, one for env and one for just the room
        # this allows us to add other prims to environment for randomization and still hide them all at once
        self._environment = stage.DefinePrim("/environment", "Xform")

        self._room = stage.DefinePrim("/environment/room", "Xform")

        from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return False
        self._asset_path = nucleus_server + "/Isaac"
        stage_path = self._asset_path + "/Environments/Simple_Room/simple_room.usd"

        self._room.GetReferences().AddReference(stage_path)

        self._target_prim = self.kit.create_prim(
            "/objects/cube", "Cube", translation=(0, 0, 100), scale=(10, 10, 50), semantic_label="target"
        )
        # make sure that we wait for the stage to load
        self.kit.app.update()
        self.kit.app.update()
        return True

    def create_camera(self):
        self._camera = self.kit.create_prim(
            "/World/Camera",
            "Camera",
            translation=(0.0, 0.0, 0.0),
            attributes={
                "focusDistance": FOCUS_DIST,
                "focalLength": FOCAL_LEN,
                "horizontalAperture": HORIZONTAL_APERTURE,
                "verticalAperture": VERTICAL_APERTURE,
            },
        )

        # activate new camera
        self._viewport.get_viewport_window().set_active_camera(str(self._camera.GetPath()))

        # the camera reference frame between sdk and sim seems to be flipped 180 on x
        # this prim acts as a proxy to do that coordinate transformation
        self._camera_proxy = self.kit.create_prim("/World/Camera/proxy", "Xform", rotation=(180, 0, 0))

    def create_bridge_components(self):
        result, self.occluded_provider = omni.kit.commands.execute(
            "RobotEngineBridgeCreateCamera",
            path="/World/REB_Occluded_Provider",
            parent=None,
            rgb_output_component="output",
            rgb_output_channel="encoder_color",
            depth_output_component="output",
            depth_output_channel="encoder_depth",
            segmentation_output_component="output",
            segmentation_output_channel="encoder_segmentation",
            bbox2d_output_component="output",
            bbox2d_output_channel="encoder_bbox",
            bbox2d_class_list="",
            bbox3d_output_component="output",
            bbox3d_output_channel="encoder_bbox3d",
            bbox3d_class_list="",
            rgb_enabled=True,
            depth_enabled=False,
            segmentaion_enabled=True,
            bbox2d_enabled=False,
            bbox3d_enabled=False,
            camera_prim_rel=[self._camera.GetPath()],
            resolution=self.Gf.Vec2i(1280, 720),
        )

        result, self.unoccluded_provider = omni.kit.commands.execute(
            "RobotEngineBridgeCreateCamera",
            path="/World/REB_Unoccluded_Provider",
            parent=None,
            rgb_output_component="output",
            rgb_output_channel="decoder_color",
            depth_output_component="output",
            depth_output_channel="decoder_depth",
            segmentation_output_component="output",
            segmentation_output_channel="decoder_segmentation",
            bbox2d_output_component="output",
            bbox2d_output_channel="decoder_bbox",
            bbox2d_class_list="",
            bbox3d_output_component="output",
            bbox3d_output_channel="decoder_bbox3d",
            bbox3d_class_list="",
            rgb_enabled=True,
            depth_enabled=False,
            segmentaion_enabled=True,
            bbox2d_enabled=False,
            bbox3d_enabled=False,
            camera_prim_rel=[self._camera.GetPath()],
            resolution=self.Gf.Vec2i(1280, 720),
        )

        # turn both cameras off so that we don't send an image when time is stepped
        self.occluded_provider.GetEnabledAttr().Set(False)
        self.unoccluded_provider.GetEnabledAttr().Set(False)

        # create rigid body sink to publish ground truth pose information
        result, self.rbs_provider = omni.kit.commands.execute(
            "RobotEngineBridgeCreateRigidBodySink",
            path="/World/REB_RigidBodiesSink",
            parent=None,
            enabled=False,
            output_component="output",
            output_channel="bodies",
            rigid_body_prims_rel=[self._camera_proxy.GetPath(), self._target_prim.GetPath()],
        )
        # disable rigid body sink until the final image is sent out so its only published once
        self.rbs_provider.GetEnabledAttr().Set(False)

    def configure_bridge(self, json_file: str = "isaacsim.app.json"):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("omni.isaac.robot_engine_bridge")
        reb_extension_path = ext_manager.get_extension_path(ext_id)
        app_file = f"{reb_extension_path}/resources/isaac_engine/json/{json_file}"
        carb.log_info(f"create application with: {reb_extension_path} {app_file}")
        return omni.kit.commands.execute(
            "RobotEngineBridgeCreateApplication", asset_path=reb_extension_path, app_file=app_file
        )

    def configure_randomization(self):
        texture_list = [
            self._asset_path + "/Samples/DR/Materials/Textures/checkered.png",
            self._asset_path + "/Samples/DR/Materials/Textures/marble_tile.png",
            self._asset_path + "/Samples/DR/Materials/Textures/picture_a.png",
            self._asset_path + "/Samples/DR/Materials/Textures/picture_b.png",
            self._asset_path + "/Samples/DR/Materials/Textures/textured_wall.png",
            self._asset_path + "/Samples/DR/Materials/Textures/checkered_color.png",
        ]
        base_path = str(self._room.GetPath())
        self.texture_comp = self.dr_helper.create_texture_comp([base_path], False, texture_list)
        # self.color_comp = self.dr_helper.create_color_comp([base_path+"/floor"])
        # disable automatic DR, we run it ourselves in the step function

        # add a movement and rotation component
        # the movement component is offset by 100cm in z so that the object remains above the table
        self.movement_comp = self.dr_helper.create_movement_comp(
            [str(self._target_prim.GetPath())], min_range=(-10, -10, -10 + 100), max_range=(10, 10, 10 + 100)
        )
        self.rotation_comp = self.dr_helper.create_rotation_comp([str(self._target_prim.GetPath())])

        self.dr_helper.toggle_manual_mode()

    def randomize_camera(self):
        # randomize camera position
        self._viewport.get_viewport_window().set_camera_position(
            str(self._camera.GetPath()),
            random.randrange(-250, 250),
            random.randrange(-250, 250),
            random.randrange(10, 250),
            True,
        )

        # get target pose and point camera at it
        pose = omni.usd.get_world_transform_matrix(self._target_prim)
        # can specify an offset on target position
        target = pose.ExtractTranslation() + self.Gf.Vec3d(0, 0, 0)

        self._viewport.get_viewport_window().set_camera_target(
            str(self._camera.GetPath()), target[0], target[1], target[2], True
        )

    def randomize_scene(self):
        self.dr_helper.randomize_once()

    def toggle_environment(self, state):
        imageable = self.UsdGeom.Imageable(self._environment)
        if state:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def step(self):
        # randomize camera every frame
        self.randomize_camera()
        # randomize textures every 10 frames
        if self.frame % RANDOMIZE_SCENE_EVERY_N_STEPS == 0:
            self.randomize_scene()

        self.toggle_environment(True)
        self.kit.update(1.0 / 60.0)
        # render occluded view
        omni.kit.commands.execute("RobotEngineBridgeTickComponent", path=str(self.occluded_provider.GetPath()))
        # hide everything but the object
        self.toggle_environment(False)
        self.kit.update(0)
        # render unoccluded view
        omni.kit.commands.execute("RobotEngineBridgeTickComponent", path=str(self.unoccluded_provider.GetPath()))
        omni.kit.commands.execute("RobotEngineBridgeTickComponent", path=str(self.rbs_provider.GetPath()))
        # output fps every 100 frames
        if self.frame % 100 == 0:
            print("FPS: ", self._viewport.get_viewport_window().get_fps())
        self.frame = self.frame + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Occluded and Unoccluded data")
    parser.add_argument("--test", action="store_true")
    args, unknown = parser.parse_known_args()
    sample = DualCameraSample()
    # On start if state creation was successful
    if sample.create_stage():
        sample.create_camera()
        sample.configure_randomization()
        # wait for stage to load
        while sample.kit.is_loading():
            sample.kit.update(0)

        sample.create_bridge_components()
        sample.configure_bridge()

        sample.start()

        while sample.kit.app.is_running():
            sample.step()
            if args.test:
                break
        sample.stop()
        sample.shutdown()
