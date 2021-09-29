# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
import omni
from pxr import UsdGeom, Gf

import numpy as np


class Jetracer:
    def __init__(self, omni_kit):
        from omni.isaac.dynamic_control import _dynamic_control
        from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

        self.omni_kit = omni_kit

        # Enable this after stage is loaded to prevent errors
        ext_manager = self.omni_kit.app.get_extension_manager()
        ext_manager.set_extension_enabled("omni.physx.vehicle", True)

        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        self.usd_path = nucleus_server + "/Isaac/Robots/Jetracer/jetracer.usd"
        self.robot_prim = None
        self._dynamic_control = _dynamic_control
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.ar = None

    # rotation is in degrees
    def spawn(self, location, rotation):
        stage = self.omni_kit.get_stage()
        prefix = "/World/Robot/Jetracer"
        prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
        print(prim_path)
        self.robot_prim = stage.DefinePrim(prim_path, "Xform")
        self.robot_prim.GetReferences().AddReference(self.usd_path)
        xform = UsdGeom.Xformable(self.robot_prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        mat = Gf.Matrix4d().SetTranslate(location)
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation))
        xform_op.Set(mat)

        self.camera_path = prim_path + "/Jetracer/Vehicle/jetracer_camera"
        # self.camera_path = prim_path + "Vehicle/jetracer_camera"

    def teleport(self, location, rotation, settle=False):
        if self.ar is None:
            self.ar = self.dc.get_rigid_body(self.robot_prim.GetPath().pathString + "/Vehicle")
            self.chassis = self.ar
        self.dc.wake_up_rigid_body(self.ar)
        rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation).GetQuaternion()

        tf = self._dynamic_control.Transform(
            location,
            (rot_quat.GetImaginary()[0], rot_quat.GetImaginary()[1], rot_quat.GetImaginary()[2], rot_quat.GetReal()),
        )
        self.dc.set_rigid_body_pose(self.chassis, tf)
        self.dc.set_rigid_body_linear_velocity(self.chassis, [0, 0, 0])
        self.dc.set_rigid_body_angular_velocity(self.chassis, [0, 0, 0])
        self.command((0, 0))
        if settle:
            frame = 0
            velocity = 1
            print("Settling robot...")
            while velocity > 0.1 and frame < 120:
                self.omni_kit.update(1.0 / 60.0)
                lin_vel = self.dc.get_rigid_body_linear_velocity(self.chassis)
                velocity = np.linalg.norm([lin_vel.x, lin_vel.y, lin_vel.z])
                # print("velocity magnitude is: ", velocity)
                frame = frame + 1
            # print("done after frame: HERE", frame)

    def activate_camera(self):
        vpi = omni.kit.viewport.get_viewport_interface()
        vpi.get_viewport_window().set_active_camera(str(self.camera_path))

    def command(self, motor_value):
        if self.ar is None:
            vehicle_path = self.robot_prim.GetPath().pathString + "/Jetracer/Vehicle"
            print(vehicle_path)
            self.ar = self.dc.get_rigid_body(vehicle_path)
            self.chassis = self.ar
            print(self.chassis)

            stage = self.omni_kit.get_stage()

            # for child_prim in stage.Traverse():
            #     print(child_prim.GetPath().pathString)

            self.accelerator = stage.GetPrimAtPath(vehicle_path).GetAttribute("physxVehicleController:accelerator")
            self.left_steer = stage.GetPrimAtPath(vehicle_path).GetAttribute("physxVehicleController:steerLeft")
            self.right_steer = stage.GetPrimAtPath(vehicle_path).GetAttribute("physxVehicleController:steerRight")
            self.target_gear = stage.GetPrimAtPath(vehicle_path).GetAttribute("physxVehicleController:targetGear")
            # TODO add brake physxVehicleController:brake

        self.dc.wake_up_rigid_body(self.ar)
        accel_cmd = self.wheel_speed_from_motor_value(motor_value[0])
        steer_left_cmd = self.wheel_speed_from_motor_value(motor_value[1])

        acceleration = max(min(accel_cmd, 1), -1)
        steering = max(min(steer_left_cmd, 1), -1)

        gear = 1  # going forward
        if acceleration < 0:
            gear = -1  # reverse

        self.accelerator.Set(abs(acceleration))
        self.target_gear.Set(gear)

        if steering > 0:
            self.right_steer.Set(steering)
        else:
            self.left_steer.Set(abs(steering))

    # idealized motor model that converts a pwm value to a velocity
    def wheel_speed_from_motor_value(self, input):
        threshold = 0.05
        if input >= 0:
            if input > threshold:
                return 1.604 * input - 0.05
            else:
                return 0
        elif input < 0:
            if input < -threshold:
                return 1.725 * input + 0.0757
            else:
                return 0

    def observations(self):
        if self.ar is None:
            self.ar = self.dc.get_rigid_body(self.robot_prim.GetPath().pathString + "/Vehicle")
            self.chassis = self.ar
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
