import omni
from omni.physx.scripts import utils
import random
import omni.isaac.dr as dr
import omni.kit
import omni.usd
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf, PhysicsSchemaTools
from omni.isaac.dynamic_control import _dynamic_control
import time
import numpy as np

class Env_config:
    def __init__(self):
        self.dr_interface = dr._dr.acquire_dr_interface()
        self.usd_path = "omniverse://localhost/Isaac/Robots/Jetbot/jetbot.usd"
        self.robot_prim = None
        self._dynamic_control = _dynamic_control
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.ar = None

    def create_physicsscene_ground_plane(self):    
        stage = omni.usd.get_context().get_stage()
        # Add a physics scene prim to stage
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
        # Set gravity vector
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(981.0)
        
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/physicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physicsScene")
        physxSceneAPI.CreateEnableCCDAttr(True)
        physxSceneAPI.CreateEnableStabilizationAttr(True)
        physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
        physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
        physxSceneAPI.CreateSolverTypeAttr("TGS")
    
        stage = omni.usd.get_context().get_stage()
        PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 1000, Gf.Vec3f(0, 0, -0), Gf.Vec3f(1.0))
    
    def create_objects(self, cube_num, cylinder_num, sphere_num):
        object_list = []
        # create cube
        stage = omni.usd.get_context().get_stage()
    
        for num in range(cube_num):
            # create first cube
            result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
            if num == 0:
                object_list.append("/World/Cube")
                continue
            if num < 10:
                object_list.append("/World/Cube_0"+str(num))
            else:
                object_list.append("/World/Cube_"+str(num))
            
        # create cylinder
        for num in range(cylinder_num):
            # create first cylinder
            result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cylinder")
            if num == 0:
                object_list.append("/World/Cylinder")
                continue
            if num < 10:
                object_list.append("/World/Cylinder_0"+str(num))
            else:
                object_list.append("/World/Cylinder_"+str(num))
    
        # create sphere
        for num in range(sphere_num):
            # create first sphere
            result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Sphere")
            if num == 0:
                object_list.append("/World/Sphere")
                continue
            if num < 10:
                object_list.append("/World/Sphere_0"+str(num))
            else:
                object_list.append("/World/Sphere_"+str(num))
                
        for mesh in object_list:
            random_num_x = random.uniform(-50,50)
            random_num_y = random.uniform(-50,50)
            cube_prim = stage.GetPrimAtPath(mesh)
            UsdGeom.XformCommonAPI(cube_prim).SetTranslate([random_num_x*10.0, random_num_y*10.0, 50.0])
            UsdGeom.XformCommonAPI(cube_prim).SetRotate((0.0, 0.0, 0.0))
            UsdGeom.XformCommonAPI(cube_prim).SetScale((1.0, 1.0, 1.0))
            utils.setRigidBody(cube_prim, "convexHull", False)
            
        return object_list
    
    def domain_randomization_test(target_list):
        asset_path = "omniverse://localhost/Isaac"
        # List of textures to randomize from
        texture_list = [
            asset_path + "/Samples/DR/Materials/Textures/checkered.png",
            asset_path + "/Samples/DR/Materials/Textures/marble_tile.png",
            asset_path + "/Samples/DR/Materials/Textures/picture_a.png",
            asset_path + "/Samples/DR/Materials/Textures/picture_b.png",
            asset_path + "/Samples/DR/Materials/Textures/textured_wall.png",
            asset_path + "/Samples/DR/Materials/Textures/checkered_color.png",
        ]
        
        # domain randomization on textures
        result, prim = omni.kit.commands.execute(
            "CreateTextureComponentCommand",
            prim_paths=target_list,
            enable_project_uvw=False,
            texture_list=texture_list,
            ignored_class_list=[],
            grouped_class_list=[],
            duration=1,
            include_children=False,
            seed=12345
        )
        
        # domain randomization on position
        result, prim = omni.kit.commands.execute(
            'CreateMovementComponentCommand',
	    path='/World/movement_component',
	    prim_paths=target_list,
	    min_range=(-800.0, -800.0, 50.0),
	    max_range=(800.0, 800.0, 50.0),
	    target_position=None,
	    target_paths=None,
	    duration=1,
            include_children=False,
            seed=12345)
	    
	# domain randomization on scale
        result, prim = omni.kit.commands.execute(
            'CreateScaleComponentCommand',
            path='/World/scale_component',
            prim_paths=target_list,
            min_range=(0.5, 0.5, 1),
            max_range=(2.0, 2.0, 1),
            uniform_scaling=False,
            duration=1,
            include_children=False,
            seed=12345)
