# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.python_app import OmniKitHelper
import random

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": True,
}

if __name__ == "__main__":
    # Simple example showing how to change resolution
    kit = OmniKitHelper(config=CONFIG)
    kit.update(1.0 / 60.0)
    for i in range(100):
        width = random.randint(128, 1980)
        height = random.randint(128, 1980)
        kit.set_setting("/app/renderer/resolution/width", width)
        kit.set_setting("/app/renderer/resolution/height", height)
        kit.update(1.0 / 60.0)
        print(f"resolution set to: {width}, {height}")

    # cleanup
    kit.shutdown()
