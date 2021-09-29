# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import carb
import omni.kit.app
import asyncio

# Simple example showing the minimal setup to run omniverse app from python


class PythonApp:
    def __init__(self):
        # Load app plugin
        self.framework = carb.get_framework()
        self.framework.load_plugins(
            loaded_file_wildcards=["omni.kit.app.plugin"],
            search_paths=[os.path.abspath(f'{os.environ["CARB_APP_PATH"]}/plugins')],
        )
        self.app = omni.kit.app.get_app()

        # Path to where kit was built to
        app_root = os.environ["CARB_APP_PATH"]

        # Inject experience config:
        sys.argv.insert(1, f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')

        # Add paths to extensions
        sys.argv.append(f"--ext-folder")
        sys.argv.append(f'{os.path.abspath(os.environ["ISAAC_PATH"])}/exts')
        # Run headless
        sys.argv.append("--no-window")

        # Set some settings
        sys.argv.append("--/app/asyncRendering=False")

        # Start the app
        self.app.startup("Isaac-Sim", app_root, sys.argv)

    def shutdown(self):
        # Shutdown
        self.app.shutdown()
        self.framework.unload_all_plugins()
        print("Shutdown complete")


if __name__ == "__main__":
    kit = PythonApp()

    # Do something, in this case we wait for stage to open and then exit
    stage_task = asyncio.ensure_future(omni.usd.get_context().new_stage_async())

    while not stage_task.done():
        kit.app.update()

    kit.shutdown()
