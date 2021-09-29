# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
import omni
import random
from pxr import UsdGeom, Gf, Sdf, UsdPhysics

from omni.isaac.synthetic_utils import DomainRandomization
from gtc2020_track_utils import *


class Environment:
    def __init__(self, omni_kit, z_height=0):
        from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

        self.omni_kit = omni_kit
        self.find_nucleus_server = find_nucleus_server
        result, nucleus_server = self.find_nucleus_server()
        if result is False:
            carb.log_error(
                "Could not find nucleus server with /Isaac folder. Please specify the correct nucleus server in apps/omni.isaac.sim.python.kit"
            )
            return

        self.texture_list = [
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/checkered.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/picture_a.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/picture_b.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/textured_wall.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/checkered_color.png",
        ]

        self.prims = []  # list of spawned tiles
        self.height = z_height  # height of the ground tiles
        self.state = None
        # because the ground plane is what the robot drives on, we only do this once. We can then re-generate the road as often as we need without impacting physics
        self.setup_physics()

        contents = omni.client.list(nucleus_server + "/Isaac/Props/Sortbot_Housing/Materials/Textures/")[1]
        for entry in contents:
            self.texture_list.append(
                nucleus_server + "/Isaac/Props/Sortbot_Housing/Materials/Textures/" + entry.relative_path
            )

        contents = omni.client.list(nucleus_server + "/Isaac/Props/YCB/Axis_Aligned/")[1]
        names = []
        loaded_paths = []

        for entry in contents:
            if not entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                names.append(nucleus_server + "/Isaac/Props/YCB/Axis_Aligned/" + entry.relative_path)
                loaded_paths.append("/World/DR/mesh_component/mesh_" + entry.relative_path[0:-4])
        print(loaded_paths)

        self.omni_kit.create_prim("/World/Floor", "Xform")

        stage = omni.usd.get_context().get_stage()
        cubeGeom = UsdGeom.Cube.Define(stage, "/World/Floor/thefloor")
        cubeGeom.CreateSizeAttr(300)
        offset = Gf.Vec3f(75, 75, -150.1)
        cubeGeom.AddTranslateOp().Set(offset)

        prims = []
        self.dr = DomainRandomization()
        self.dr.toggle_manual_mode()
        self.dr.create_mesh_comp(prim_paths=prims, mesh_list=names, mesh_range=[1, 1])
        self.omni_kit.update(1 / 60.0)
        print("waiting for materials to load...")

        while self.omni_kit.is_loading():
            self.omni_kit.update(1 / 60.0)

        lights = []
        for i in range(5):
            prim_path = "/World/Lights/light_" + str(i)
            self.omni_kit.create_prim(
                prim_path,
                "SphereLight",
                translation=(0, 0, 200),
                rotation=(0, 0, 0),
                attributes={"radius": 10, "intensity": 1000.0, "color": (1.0, 1.0, 1.0)},
            )
            lights.append(prim_path)

        self.dr.create_movement_comp(
            prim_paths=loaded_paths, min_range=(0, 0, 15), max_range=(TRACK_DIMS[0], TRACK_DIMS[1], 15)
        )
        self.dr.create_rotation_comp(prim_paths=loaded_paths)
        self.dr.create_visibility_comp(prim_paths=loaded_paths, num_visible_range=(15, 15))

        self.dr.create_light_comp(light_paths=lights)
        self.dr.create_movement_comp(
            prim_paths=lights, min_range=(0, 0, 30), max_range=(TRACK_DIMS[0], TRACK_DIMS[1], 30)
        )

        self.dr.create_texture_comp(
            prim_paths=["/World/Floor"], enable_project_uvw=True, texture_list=self.texture_list
        )

    def generate_lights(self):
        # TODO: center this onto the track
        prim_path = omni.usd.get_stage_next_free_path(self.omni_kit.get_stage(), "/World/Env/Light", False)
        # self.prims.append(prim_path)
        # LOCMOD revisit (don't add so it won't be removed on reset)
        self.omni_kit.create_prim(
            prim_path,
            "RectLight",
            translation=(75, 75, 100),
            rotation=(0, 0, 0),
            attributes={"height": 150, "width": 150, "intensity": 2000.0, "color": (1.0, 1.0, 1.0)},
        )

    def reset(self, shape):
        # this deletes objects in self.prims
        stage = omni.usd.get_context().get_stage()
        for layer in stage.GetLayerStack():
            edit = Sdf.BatchNamespaceEdit()
            for path in self.prims:
                prim_spec = layer.GetPrimAtPath(path)
                if prim_spec is None:
                    continue
                parent_spec = prim_spec.realNameParent
                if parent_spec is not None:
                    edit.Add(path, Sdf.Path.emptyPath)
            layer.Apply(edit)

        self.prims = []

        # self.pxrImageable.MakeInvisible()
        # LOCMOD revisit
        # self.generate_road(shape)
        self.dr.randomize_once()

    def generate_road(self, shape):
        stage = self.omni_kit.get_stage()
        self.add_track(stage)

    def add_track(self, stage):
        result, nucleus_server = self.find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        path = nucleus_server + "/Isaac/Environments/Jetracer/jetracer_track_solid.usd"
        prefix = "/World/Env/Track"
        prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
        # self.prims.append(prim_path) #(don't add so the jetracer track won't be removed on reset)
        track_prim = stage.DefinePrim(prim_path, "Xform")
        track_prim.GetReferences().AddReference(path)
        # xform = UsdGeom.Xformable(track_prim)
        # xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        # mat = Gf.Matrix4d().SetTranslate(location)
        # mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation))
        # xform_op.Set(mat)

    def setup_physics(self):
        from pxr import PhysxSchema, PhysicsSchemaTools

        stage = self.omni_kit.get_stage()
        # Add physics scene
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/Env/PhysicsScene"))
        # Set gravity vector
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(981.0)
        # Set physics scene to use cpu physics
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/Env/PhysicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/Env/PhysicsScene")
        physxSceneAPI.CreateEnableCCDAttr(True)
        physxSceneAPI.CreateEnableStabilizationAttr(True)
        physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
        physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
        physxSceneAPI.CreateSolverTypeAttr("TGS")
        # Create physics plane for the ground
        PhysicsSchemaTools.addGroundPlane(
            stage, "/World/Env/GroundPlane", "Z", 100.0, Gf.Vec3f(0, 0, self.height), Gf.Vec3f(1.0)
        )
        # Hide the visual geometry
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Env/GroundPlane/geom"))
        if imageable:
            imageable.MakeInvisible()

    def get_valid_location(self):
        # keep try until within the center track
        dist = 1
        x = 4
        y = 4
        while dist > LANE_WIDTH:
            x = random.randint(0, TRACK_DIMS[0])
            y = random.randint(0, TRACK_DIMS[1])
            dist = center_line_dist(np.array([x, y]))

        print("get valid location called", x, y)
        return (x, y)
