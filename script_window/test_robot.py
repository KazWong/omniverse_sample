import omni
from pxr import UsdGeom, Gf, UsdPhysics
import numpy as np

# Camera parameters
FOCAL_LENGTH = 0.75
HORIZONTAL_APERTURE = 2.350
VERTICAL_APERTURE = 2.350

# Drive Parameters
DRIVE_STIFFNESS = 10000.0
# The amount the camera points down at, decrease to raise camera angle
CAMERA_PIVOT = 40.0
from omni.isaac.dynamic_control import _dynamic_control
        
class Robot_config:
    def __init__(self):
        self.usd_path = "omniverse://localhost/Library/Robots/config_robot/robot_event_cam.usd"
        self.robot_prim = None
        self._dynamic_control = _dynamic_control
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.ar = None
        print("init", self.ar)

    # rotation is in degrees
    def spawn(self, location, rotation):
        stage = omni.usd.get_context().get_stage()
        prefix = "/World/robot_event_cam"
        prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
        self.robot_prim = stage.DefinePrim(prim_path, "Xform")
        self.robot_prim.GetReferences().AddReference(self.usd_path)
        xform = UsdGeom.Xformable(self.robot_prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        mat = Gf.Matrix4d().SetTranslate(location)
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation))
        xform_op.Set(mat)
        return
        #self.camera_path = prim_path + "/chassis/rgb_camera/jetbot_camera"
        #self.camera_pivot = prim_path + "/chassis/rgb_camera"

        # Set joint drive parameters
        wheel_back_left_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_back_left_joint"), "angular")
        wheel_back_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)

        wheel_back_right_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_back_right_joint"), "angular")
        wheel_back_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
        wheel_front_left_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_front_left_joint"), "angular")
        wheel_front_left_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)
        
        wheel_front_right_joint = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/agv_base_link/wheel_front_right_joint"), "angular")
        wheel_front_right_joint.GetDampingAttr().Set(DRIVE_STIFFNESS)

    def teleport(self, location, rotation, settle=False):
        stage = omni.usd.get_context().get_stage()
        prefix = "/World/Robot/Jetbot"
        prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
        self.robot_prim = stage.DefinePrim(prim_path, "Xform")
        self.robot_prim.GetReferences().AddReference(self.usd_path)
        print("before teleport", self.ar)
        if self.ar is None:
            print(type(self.robot_prim.GetPath().pathString), self.robot_prim.GetPath().pathString)
            self.ar = self.dc.get_articulation(self.robot_prim.GetPath().pathString)
            print("after teleport", self.ar)
            self.chassis = self.dc.get_articulation_root_body(self.ar)

        self.dc.wake_up_articulation(self.ar)
        rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation).GetQuaternion()

        tf = self._dynamic_control.Transform(
            location,
            (rot_quat.GetImaginary()[0], rot_quat.GetImaginary()[1], rot_quat.GetImaginary()[2], rot_quat.GetReal()),
        )
        self.dc.set_rigid_body_pose(self.chassis, tf)
        self.dc.set_rigid_body_linear_velocity(self.chassis, [0, 0, 0])
        self.dc.set_rigid_body_angular_velocity(self.chassis, [0, 0, 0])
        self.command((0, 0))
        # Settle the robot onto the ground
        if settle:
            frame = 0
            velocity = 1
            while velocity > 0.1 and frame < 120:
                omni.usd.get_context().update(1.0 / 60.0)
                lin_vel = self.dc.get_rigid_body_linear_velocity(self.chassis)
                velocity = np.linalg.norm([lin_vel.x, lin_vel.y, lin_vel.z])
                frame = frame + 1

    def activate_camera(self):
        # Set camera parameters
        stage = omni.usd.get_context().get_stage()
        cameraPrim = UsdGeom.Camera(stage.GetPrimAtPath(self.camera_path))
        cameraPrim.GetFocalLengthAttr().Set(FOCAL_LENGTH)
        cameraPrim.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE)
        cameraPrim.GetVerticalApertureAttr().Set(VERTICAL_APERTURE)

        # Point camera down at road
        pivot_prim = stage.GetPrimAtPath(self.camera_pivot)
        transform_attr = pivot_prim.GetAttribute("xformOp:transform")
        transform_attr.Set(
            transform_attr.Get().SetRotateOnly(Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(0, 1, 0), CAMERA_PIVOT)))
        )

        vpi = omni.kit.viewport.get_viewport_interface()
        vpi.get_viewport_window().set_active_camera(str(self.camera_path))

    def command(self, motor_value):
        print("*"*50)
        print("running command")
        print(self.ar)
        if self.ar is None:
            self.ar = self.dc.get_articulation(self.robot_prim.GetPath().pathString)
            self.chassis = self.dc.get_articulation_root_body(self.ar)
            print("here")
            self.wheel_left = self.dc.find_articulation_dof(self.ar, "left_wheel_joint")
            self.wheel_right = self.dc.find_articulation_dof(self.ar, "right_wheel_joint")
            print(self.wheel_left)
            print("-----------")
            print(self.wheel_right)
        self.dc.wake_up_articulation(self.ar)
        left_speed = self.wheel_speed_from_motor_value(motor_value[0])
        right_speed = self.wheel_speed_from_motor_value(motor_value[1])
        self.dc.set_dof_velocity_target(self.wheel_left, np.clip(left_speed, -10, 10))
        self.dc.set_dof_velocity_target(self.wheel_right, np.clip(right_speed, -10, 10))

    # idealized motor model
    def wheel_speed_from_motor_value(self, input):
        return input

    def observations(self):
        if self.ar is None:
            self.ar = self.dc.get_articulation(self.robot_prim.GetPath().pathString)
            self.chassis = self.dc.get_articulation_root_body(self.ar)
        dc_pose = self.dc.get_rigid_body_pose(self.chassis)
        dc_lin_vel = self.dc.get_rigid_body_linear_velocity(self.chassis)
        dc_local_lin_vel = self.dc.get_rigid_body_local_linear_velocity(self.chassis)
        dc_ang_vel = self.dc.get_rigid_body_angular_velocity(self.chassis)
        return {
            "pose": (dc_pose.p.x, dc_pose.p.y, dc_pose.p.z, dc_pose.r.w, dc_pose.r.x, dc_pose.r.y, dc_pose.r.z),
            "linear_velocity": (dc_lin_vel.x, dc_lin_vel.y, dc_lin_vel.z),
            "local_linear_velocity": (dc_local_lin_vel.x, dc_local_lin_vel.y, dc_local_lin_vel.z),
            "angular_velocity": (dc_ang_vel.x, dc_ang_vel.y, dc_ang_vel.z),
        }
