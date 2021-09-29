# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Demonstration of using OmniKit to generate a scene, collect groundtruth and visualize
the results. This advanced sample also simulates physics and uses a custom glass material
"""

import copy
import os
import random
import numpy as np
from omni.isaac.python_app import OmniKitHelper
import matplotlib.pyplot as plt

TRANSLATION_RANGE = 300.0
SCALE = 50.0

# specify a custom config
CUSTOM_CONFIG = {
    "width": 1024,
    "height": 1024,
    "renderer": "PathTracing",
    "samples_per_pixel_per_frame": 128,
    "max_bounces": 10,
    "max_specular_transmission_bounces": 6,
    "max_volume_bounces": 4,
    "subdiv_refinement_level": 2,
    "headless": True,
    "sync_loads": True,
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
}


def main():
    kit = OmniKitHelper(CUSTOM_CONFIG)
    from pxr import Gf, Sdf, UsdShade, UsdGeom, Semantics
    from omni.isaac.synthetic_utils import SyntheticDataHelper

    sd_helper = SyntheticDataHelper()

    from omni.syntheticdata import visualize, helpers
    from omni.physx.scripts import utils
    from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools
    import omni

    # SCENE SETUP
    # Get the current stage
    stage = kit.get_stage()

    # Add a sphere light
    kit.create_prim(
        "/World/Light1",
        "SphereLight",
        translation=(0, 200, 0),
        attributes={"radius": 100, "intensity": 100000.0, "color": (1, 1, 1)},
    )

    # Add physics scene
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    # Set gravity vector
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(981.0)
    # Set physics scene to use cpu physics
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/physicsScene"))
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physicsScene")
    physxSceneAPI.CreateEnableCCDAttr(True)
    physxSceneAPI.CreateEnableStabilizationAttr(True)
    physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
    physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
    physxSceneAPI.CreateSolverTypeAttr("TGS")

    # Create a ground plane
    PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 1000, Gf.Vec3f(0, 0, -100), Gf.Vec3f(1.0))

    # Create 10 randomly positioned and coloured spheres and cube
    # We will assign each a semantic label based on their shape (sphere/cube/cone)
    prims = []
    for i in range(10):
        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        prim = stage.DefinePrim(f"/World/cube{i}", prim_type)
        translation = np.random.rand(3) * TRANSLATION_RANGE
        UsdGeom.XformCommonAPI(prim).SetTranslate(translation.tolist())
        UsdGeom.XformCommonAPI(prim).SetScale((SCALE, SCALE, SCALE))
        # prim.GetAttribute("primvars:displayColor").Set([np.random.rand(3).tolist()])

        # Add semantic label based on prim type
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set(prim_type)

        # Add physics to prims
        utils.setRigidBody(prim, "convexHull", False)
        # Set Mass to 1 kg
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(1)
        # add prim reference to list
        prims.append(prim)

    # Apply glass material
    for prim in prims:
        # Create Glass material
        mtl_created_list = []
        kit.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_created_list=mtl_created_list,
        )
        mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])

        # Set material inputs, these can be determined by looking at the .mdl file
        # or by selecting the Shader attached to the Material in the stage window and looking at the details panel
        color = Gf.Vec3f(random.random(), random.random(), random.random())
        omni.usd.create_material_input(mtl_prim, "glass_color", color, Sdf.ValueTypeNames.Color3f)
        omni.usd.create_material_input(mtl_prim, "glass_ior", 1.25, Sdf.ValueTypeNames.Float)
        # This value is the volumetric light absorption scale, reduce to zero to make glass clearer
        omni.usd.create_material_input(mtl_prim, "depth", 0.001, Sdf.ValueTypeNames.Float)
        # Enable for thin glass objects if needed
        omni.usd.create_material_input(mtl_prim, "thin_walled", False, Sdf.ValueTypeNames.Bool)
        # Bind the material to the prim
        prim_mat_shade = UsdShade.Material(mtl_prim)
        UsdShade.MaterialBindingAPI(prim).Bind(prim_mat_shade, UsdShade.Tokens.strongerThanDescendants)

    # force RayTracedLighting mode for better performance while simulating physics
    kit.set_setting("/rtx/rendermode", "RayTracedLighting")

    # start simulation
    kit.play()
    # Step simulation so that objects fall to rest
    # wait until all materials are loaded
    frame = 0
    print("simulating physics...")
    while frame < 60 or kit.is_loading():
        kit.update(1 / 60.0)
        frame = frame + 1
    print("done")

    # Return to user specified render mode
    kit.set_setting("/rtx/rendermode", CUSTOM_CONFIG["renderer"])
    print("capturing...")
    # Get groundtruth using glass material
    kit.update()
    viewport = omni.kit.viewport.get_default_viewport_window()
    gt = sd_helper.get_groundtruth(
        [
            "rgb",
            "depth",
            "boundingBox2DTight",
            "boundingBox2DLoose",
            "instanceSegmentation",
            "semanticSegmentation",
            "boundingBox3D",
        ],
        viewport,
    )
    print("done")
    # everything is captured, stop simulating
    kit.stop()
    print("visualize results")
    # GROUNDTRUTH VISUALIZATION
    # Setup a figure
    _, axes = plt.subplots(2, 4, figsize=(20, 7))
    axes = axes.flat
    for ax in axes:
        ax.axis("off")

    # RGB
    axes[0].set_title("RGB")
    for ax in axes[:-1]:
        ax.imshow(gt["rgb"])

    # DEPTH
    axes[1].set_title("Depth")
    depth_data = np.clip(gt["depth"], 0, 255)
    axes[1].imshow(visualize.colorize_depth(depth_data.squeeze()))

    # BBOX2D TIGHT
    axes[2].set_title("BBox 2D Tight")
    rgb_data = copy.deepcopy(gt["rgb"])
    axes[2].imshow(visualize.colorize_bboxes(gt["boundingBox2DTight"], rgb_data))

    # BBOX2D LOOSE
    axes[3].set_title("BBox 2D Loose")
    rgb_data = copy.deepcopy(gt["rgb"])
    axes[3].imshow(visualize.colorize_bboxes(gt["boundingBox2DLoose"], rgb_data))

    # INSTANCE SEGMENTATION
    axes[4].set_title("Instance Segmentation")
    instance_seg = gt["instanceSegmentation"][0]
    instance_rgb = visualize.colorize_segmentation(instance_seg)
    axes[4].imshow(instance_rgb, alpha=0.7)

    # SEMANTIC SEGMENTATION
    axes[5].set_title("Semantic Segmentation")
    semantic_seg = gt["semanticSegmentation"]
    semantic_rgb = visualize.colorize_segmentation(semantic_seg)
    axes[5].imshow(semantic_rgb, alpha=0.7)

    # BBOX 3D
    axes[6].set_title("BBox 3D")
    bbox_3d_data = gt["boundingBox3D"]
    bboxes_3d_corners = bbox_3d_data["corners"]
    projected_corners = helpers.world_to_image(bboxes_3d_corners.reshape(-1, 3), viewport)
    projected_corners = projected_corners.reshape(-1, 8, 3)
    rgb_data = copy.deepcopy(gt["rgb"])
    bboxes3D_rgb = visualize.colorize_bboxes_3d(projected_corners, rgb_data)
    axes[6].imshow(bboxes3D_rgb)

    # Save figure
    print("saving figure to: ", os.getcwd() + "/visualize_groundtruth_physics.png")
    plt.savefig("visualize_groundtruth_physics.png")

    # cleanup
    kit.shutdown()


if __name__ == "__main__":
    main()
