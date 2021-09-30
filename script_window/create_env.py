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
from test_robot import Robot_config
from test_env import Env_config


ar = None
usd_path = "omniverse://localhost/Library/Robots/config_robot/robot_event_cam.usd"
dc = _dynamic_control.acquire_dynamic_control_interface()

robot_prim_path = "/World/robot_event_cam"

def teleport(location, rotation, settle=False):
    global ar
    global usd_path
    global dc
    global robot_prim_path
    print("before teleport", ar)
    if ar is None:
        ar = dc.get_articulation(robot_prim_path)
        print("after teleport", ar)
        chassis = dc.get_articulation_root_body(ar)

    dc.wake_up_articulation(ar)
    rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation).GetQuaternion()

    tf = _dynamic_control.Transform(
        location,
        (rot_quat.GetImaginary()[0], rot_quat.GetImaginary()[1], rot_quat.GetImaginary()[2], rot_quat.GetReal()),
    )
    dc.set_rigid_body_pose(chassis, tf)
    dc.set_rigid_body_linear_velocity(chassis, [0, 0, 0])
    dc.set_rigid_body_angular_velocity(chassis, [0, 0, 0])
    command((-4, 4, -4, 4))
    # Settle the robot onto the ground
    if settle:
        frame = 0
        velocity = 1
        while velocity > 0.1 and frame < 120:
            omni.usd.get_context().update(1.0 / 60.0)
            lin_vel = dc.get_rigid_body_linear_velocity(chassis)
            velocity = np.linalg.norm([lin_vel.x, lin_vel.y, lin_vel.z])
            frame = frame + 1
                
def command(motor_value):
    global ar
    global dc
    global robot_prim_path
    print("*"*50)
    print("running command")
    print("ar outside command",ar)
    ar = None
    if ar is None:
        ar = dc.get_articulation(robot_prim_path)
        chassis = dc.get_articulation_root_body(ar)
        num_joints = dc.get_articulation_joint_count(ar)
        num_dofs = dc.get_articulation_dof_count(ar)
        num_bodies = dc.get_articulation_body_count(ar)
        print("ar inside command",ar)
        print("chassis inside command",chassis)
        print("num joints inside command",num_joints)
        print("num dofs inside command",num_dofs)
        print("num bodies inside command",num_bodies)
                                
        wheel_back_left = dc.find_articulation_dof(ar, "wheel_back_left_joint")
        wheel_back_right = dc.find_articulation_dof(ar, "wheel_back_right_joint")
        wheel_front_left = dc.find_articulation_dof(ar, "wheel_front_left_joint")
        wheel_front_right = dc.find_articulation_dof(ar, "wheel_front_right_joint")
        print(wheel_back_left)
        print(wheel_back_right)
        print(wheel_front_left)
        print(wheel_front_right)
        print("*"*50)
    dc.wake_up_articulation(ar)
    wheel_back_left_speed = wheel_speed_from_motor_value(motor_value[0])
    wheel_back_right_speed = wheel_speed_from_motor_value(motor_value[1])
    wheel_front_left_speed = wheel_speed_from_motor_value(motor_value[2])
    wheel_front_right_speed = wheel_speed_from_motor_value(motor_value[3])
    
    dc.set_dof_velocity_target(wheel_back_left, np.clip(wheel_back_left_speed, -10, 10))
    dc.set_dof_velocity_target(wheel_back_right, np.clip(wheel_back_right_speed, -10, 10))      
    dc.set_dof_velocity_target(wheel_front_left, np.clip(wheel_front_left_speed, -10, 10))
    dc.set_dof_velocity_target(wheel_front_right, np.clip(wheel_front_right_speed, -10, 10))             

def wheel_speed_from_motor_value(input_value):
    return input_value

if __name__ == '__main__':
    # create objects: cube, cylinder, sphere
    #obj_list = create_objects(0,0,0)
    # run the program
   
    #domain_randomization_test(obj_list)  
    flag = 0

    if flag == 0:
        # create PhysicsScene and ground plane
        test_env = Env_config()
        test_env.create_physicsscene_ground_plane()
        obj_list = test_env.create_objects(2,2,2)
        
        # spawn robot
        test_robot = Robot_config()
        test_robot.spawn((0,0,5), 0)
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
        
    elif flag == 1:
        teleport((0,0,5), 0)
    #dr_interface.randomize_once()
