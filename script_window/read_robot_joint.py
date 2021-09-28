from omni.isaac.dynamic_control import _dynamic_control

dc = _dynamic_control.acquire_dynamic_control_interface()

art = dc.get_articulation("/World/wheelbarrow")
dof_states = dc.get_articulation_dof_states(art, _dynamic_control.STATE_ALL)
#print(dof_states)

back_left = dc.find_articulation_dof(art, "wheel_left_joint")
back_right = dc.find_articulation_dof(art, "wheel_right_joint")

back_left_state = dc.get_dof_state(back_left)
back_right_state = dc.get_dof_state(back_right)

#print(back_left_state.pos)
#print(back_right_state.pos)

agv_base_link = dc.find_articulation_body(art, "agv_base_link")
base_footprint = dc.find_articulation_body(art, "base_footprint")
wheel_center_link = dc.find_articulation_body(art, "wheel_center_link")

agv_base_link_state = dc.get_rigid_body_angular_velocity(agv_base_link)
base_footprint_state = dc.get_rigid_body_angular_velocity(base_footprint)
wheel_center_link_state = dc.get_rigid_body_angular_velocity(wheel_center_link)

print(agv_base_link_state)
print(base_footprint_state)
print(wheel_center_link_state)
