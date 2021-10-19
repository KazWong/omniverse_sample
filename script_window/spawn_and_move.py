import carb
import omni
import omni.kit.app
import time

from pxr import UsdGeom, Gf, Sdf

from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
#from omni.isaac.dynamic_control import _dynamic_control
from omni.physx.scripts import utils
from omni.isaac.dynamic_control import _dynamic_control

result, nucleus = find_nucleus_server()

stage = omni.usd.get_context().get_stage()
prefix = "/World/soap_odom"
prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
print(prim_path)
robot_prim = stage.DefinePrim(prim_path, "Xform")
robot_prim.GetReferences().AddReference(nucleus + "/Library/Robots/Soap_0/soap_odom.usd")

omni.timeline.get_timeline_interface().play()

print("play")

dc = _dynamic_control.acquire_dynamic_control_interface()

art = dc.get_articulation("/World/soap_odom/odom/robot")

front_left = dc.find_articulation_dof(art, "wheel_front_left_joint")
front_right = dc.find_articulation_dof(art, "wheel_front_right_joint")
back_left = dc.find_articulation_dof(art, "wheel_back_left_joint")
back_right = dc.find_articulation_dof(art, "wheel_back_right_joint")

dc.wake_up_articulation(art)

app = omni.kit.app.get_app()

while not app.is_running():
	time.sleep(1.0)

print("running")

dc.set_dof_velocity_target(front_left, -3.14)
dc.set_dof_velocity_target(front_right, 3.14)
dc.set_dof_velocity_target(back_left, -3.14)
dc.set_dof_velocity_target(back_right, 3.14)

while not app.is_running():
	app.update()

#omni.timeline.get_timeline_interface().stop()
