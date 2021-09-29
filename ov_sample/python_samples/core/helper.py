# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.python_app import OmniKitHelper

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": True,
}

if __name__ == "__main__":
    # Simple example showing how to start and stop the helper
    kit = OmniKitHelper(config=CONFIG)

    ### Perform any omniverse imports here after the helper loads ###

    kit.play()  # Start simulation
    kit.update(1.0 / 60.0)  # Render a single frame
    kit.stop()  # Stop Simulation
    kit.shutdown()  # Cleanup application
