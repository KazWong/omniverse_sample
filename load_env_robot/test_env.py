import numpy as np

class Env_config:
    def __init__(self, omni, kit):
        self.usd_path = "omniverse://localhost/Library/Robots/config_robot/robot_event_cam.usd"
        self.kit = kit
        self.omni = omni
        
    def create_objects(self, cube_num, cylinder_num, sphere_num):
        from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf, PhysicsSchemaTools
        from omni.physx.scripts import utils
        TRANSLATION_RANGE = 500.0
        object_list = []
    	# create cube
        stage = self.kit.get_stage()
    
        for num in range(cube_num):
            # create first cube
            result, path = self.omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
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
            result, path = self.omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cylinder")
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
            result, path = self.omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Sphere")
            if num == 0:
                object_list.append("/World/Sphere")
                continue
            if num < 10:
                object_list.append("/World/Sphere_0"+str(num))
            else:
                object_list.append("/World/Sphere_"+str(num))
    
        for mesh in object_list:
            translation = np.random.rand(3) * TRANSLATION_RANGE
            translation[2] = 40.0
            cube_prim = stage.GetPrimAtPath(mesh)
            UsdGeom.XformCommonAPI(cube_prim).SetTranslate(translation.tolist())
            #UsdGeom.XformCommonAPI(cube_prim).SetRotate((0.0, 0.0, 0.0))
            #UsdGeom.XformCommonAPI(cube_prim).SetScale((30.0, 30.0, 30.0))
            utils.setRigidBody(cube_prim, "convexHull", False)
            utils.setCollider(cube_prim, approximationShape="convexHull")
            
        return object_list
        
    def domain_randomization_test(self, target_list):
        import omni.isaac.dr as dr
        dr_interface = dr._dr.acquire_dr_interface()
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
        
        # domain randomization on position
        result, prim = self.omni.kit.commands.execute(
            'CreateMovementComponentCommand',
            path='/World/movement_component',
            prim_paths=target_list,
            min_range=(-600.0, -600.0, 50.0),
            max_range=(600.0, 600.0, 50.0),
            target_position=None,
            target_paths=None,
            duration=1,
            include_children=False,
            seed=12345)
            
        # domain randomization on textures
        #result, prim = self.omni.kit.commands.execute(
        #    "CreateTextureComponentCommand",
        #    path='/World/texture_component',
        #    prim_paths=target_list,
        #    enable_project_uvw=False,
        #    texture_list=texture_list,
        #    duration=1,
        #    include_children=False,
        #    seed=12345)
            
        # domain randomization on scale
        result, prim = self.omni.kit.commands.execute(
            'CreateScaleComponentCommand',
            path='/World/scale_component',
            prim_paths=target_list,
            min_range=(0.5, 0.5, 1),
            max_range=(2.0, 2.0, 1),
            uniform_scaling=False,
            duration=1,
            include_children=False,
            seed=12345)
            
        dr_interface.toggle_manual_mode()
