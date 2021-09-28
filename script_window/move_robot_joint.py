from omni.isaac.dynamic_control import _dynamic_control

dc = _dynamic_control.acquire_dynamic_control_interface()

art = dc.get_articulation("/World/soap_odom/odom/robot")

front_left = dc.find_articulation_dof(art, "wheel_front_left_joint")
front_right = dc.find_articulation_dof(art, "wheel_front_right_joint")
back_left = dc.find_articulation_dof(art, "wheel_back_left_joint")
back_right = dc.find_articulation_dof(art, "wheel_back_right_joint")

dc.wake_up_articulation(art)
dc.set_dof_velocity_target(front_left, -3.14)
dc.set_dof_velocity_target(front_right, 3.14)
dc.set_dof_velocity_target(back_left, -3.14)
dc.set_dof_velocity_target(back_right, 3.14)
