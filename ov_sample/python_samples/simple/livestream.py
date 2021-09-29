# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.python_app import OmniKitHelper
import omni

# This sample enables a livestream server to connect to when running headless
CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "renderer": "RayTracedLighting",
    "display_options": 3807,  # Set display options to show default grid
}

if __name__ == "__main__":
    # Start the omniverse application
    kit = OmniKitHelper(config=CONFIG)

    # Enable Livestream extension
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    kit.set_setting("/app/window/drawMouse", True)
    kit.set_setting("/app/livestream/proto", "ws")
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.core", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.native", True)

    # Run until closed
    while kit.app.is_running():
        # Run in realtime mode, we don't specify the step size
        kit.update()

    kit.stop()
    kit.shutdown()
