# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import carb
from omni.isaac.python_app import OmniKitHelper

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": True,
}

if __name__ == "__main__":
    # This sample loads an articulation and prints its information
    kit = OmniKitHelper(config=CONFIG)
    import omni
    from omni.isaac.dynamic_control import _dynamic_control
    from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

    stage = kit.get_stage()

    result, nucleus_server = find_nucleus_server()
    if result is False:
        carb.log_error("Could not find nucleus server with /Isaac folder")
    asset_path = nucleus_server + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
    omni.usd.get_context().open_stage(asset_path)
    # start simulation
    kit.play()

    # perform timestep
    kit.update(1.0 / 60.0)

    dc = _dynamic_control.acquire_dynamic_control_interface()
    # Get handle to articulation
    art = dc.get_articulation("/panda")
    if art == _dynamic_control.INVALID_HANDLE:
        print("*** '%s' is not an articulation" % "/panda")
    else:
        # Print information about articulation
        root = dc.get_articulation_root_body(art)
        print(str("Got articulation handle %d \n" % art) + str("--- Hierarchy\n"))

        body_states = dc.get_articulation_body_states(art, _dynamic_control.STATE_ALL)
        print(str("--- Body states:\n") + str(body_states) + "\n")

        dof_states = dc.get_articulation_dof_states(art, _dynamic_control.STATE_ALL)
        print(str("--- DOF states:\n") + str(dof_states) + "\n")

        dof_props = dc.get_articulation_dof_properties(art)
        print(str("--- DOF properties:\n") + str(dof_props) + "\n")

    # Simulate robot coming to a rest configuration
    for i in range(100):
        kit.update(1.0 / 60.0)

    # Simulate robot for a fixed number of frames and specify a joint position target
    for i in range(100):
        dof_ptr = dc.find_articulation_dof(art, "panda_joint2")
        # This should be called each frame of simulation if state on the articulation is being changed.
        dc.wake_up_articulation(art)
        # Set joint position target
        dc.set_dof_position_target(dof_ptr, -1.5)
        kit.update(1.0 / 60.0)

    kit.stop()
    kit.shutdown()
