# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.python_app import OmniKitHelper

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": True,
}

if __name__ == "__main__":
    # URDF import, configuration and simualtion sample
    kit = OmniKitHelper(config=CONFIG)
    import omni.kit.commands
    from pxr import Sdf, Gf, UsdPhysics, UsdLux, PhysxSchema

    # Setting up import configuration:
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False

    # Get path to extension data:
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_id = ext_manager.get_enabled_extension_id("omni.isaac.urdf")
    extension_path = ext_manager.get_extension_path(ext_id)
    # Import URDF
    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=extension_path + "/data/urdf/robots/carter/urdf/carter.urdf",
        import_config=import_config,
    )
    # Get stage handle
    stage = omni.usd.get_context().get_stage()

    # Enable physics
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    # Set gravity
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(981.0)
    # Set solver settings
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
    physxSceneAPI.CreateEnableCCDAttr(True)
    physxSceneAPI.CreateEnableStabilizationAttr(True)
    physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
    physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
    physxSceneAPI.CreateSolverTypeAttr("TGS")

    # Add ground plane
    omni.kit.commands.execute(
        "AddGroundPlaneCommand",
        stage=stage,
        planePath="/groundPlane",
        axis="Z",
        size=1500.0,
        position=Gf.Vec3f(0, 0, -50),
        color=Gf.Vec3f(0.5),
    )

    # Add lighting
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(500)

    # Get handle to the Drive API for both wheels
    left_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/left_wheel"), "angular")
    right_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/right_wheel"), "angular")

    # Set the velocity drive target in degrees/second
    left_wheel_drive.GetTargetVelocityAttr().Set(150)
    right_wheel_drive.GetTargetVelocityAttr().Set(150)

    # Set the drive damping, which controls the strength of the velocity drive
    left_wheel_drive.GetDampingAttr().Set(15000)
    right_wheel_drive.GetDampingAttr().Set(15000)

    # Set the drive stiffness, which controls the strength of the position drive
    # In this case because we want to do velocity control this should be set to zero
    left_wheel_drive.GetStiffnessAttr().Set(0)
    right_wheel_drive.GetStiffnessAttr().Set(0)

    # Start simulation
    kit.play()

    # perform simulation
    for frame in range(100):
        kit.update(1.0 / 60.0)

    # Shutdown and exit
    kit.stop()
    kit.shutdown()
