# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.python_app import OmniKitHelper
import carb
import omni

# This sample loads a usd stage and starts simulation
CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": False,
    "renderer": "RayTracedLighting",
}

if __name__ == "__main__":
    import argparse

    # Set up command line arguments
    parser = argparse.ArgumentParser("Usd Load sample")
    parser.add_argument("--usd_path", type=str, help="Path to usd file", required=True)
    parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
    parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

    args, unknown = parser.parse_known_args()
    # Start the omniverse application
    CONFIG["headless"] = args.headless
    kit = OmniKitHelper(config=CONFIG)

    # Locate /Isaac folder on nucleus server to load sample
    from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

    result, nucleus_server = find_nucleus_server()
    if result is False:
        carb.log_error("Could not find nucleus server with /Isaac folder, exiting")
        exit()
    asset_path = nucleus_server + "/Isaac"
    usd_path = asset_path + args.usd_path
    omni.usd.get_context().open_stage(usd_path, None)
    # Wait two frames so that stage starts loading
    kit.app.update()
    kit.app.update()

    print("Loading stage...")
    while kit.is_loading():
        kit.update(1.0 / 60.0)
    print("Loading Complete")
    kit.play()
    # Run in test mode, exit after a fixed number of steps
    if args.test is True:
        for i in range(10):
            # Run in realtime mode, we don't specify the step size
            kit.update()
    else:
        while kit.app.is_running():
            # Run in realtime mode, we don't specify the step size
            kit.update()

    kit.stop()
    kit.shutdown()
