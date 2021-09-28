from omni.isaac.dynamic_control import _dynamic_control
dc = _dynamic_control.acquire_dynamic_control_interface()

cube = dc.get_rigid_body(f"/World/cube1")

dc.wake_up_rigid_body(cube)

tf = _dynamic_control.Transform( (250.0, 250.0, 500.0), (0.0, 0.0, 0.0, 1.0))
dc.set_rigid_body_pose(cube, tf)
