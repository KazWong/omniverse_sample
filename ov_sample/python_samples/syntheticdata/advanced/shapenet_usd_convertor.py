# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import pprint
from omni.isaac.python_app import OmniKitHelper


"""Convert ShapeNetCore V2 to USD without materials.
By only converting the ShapeNet geometry, we can more quickly load assets into scenes for the purpose of creating
large datasets or for online training of Deep Learning models.
"""

if __name__ == "__main__":
    RENDER_CONFIG = {"experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit'}
    kit = OmniKitHelper(config=RENDER_CONFIG)

    import argparse
    from omni.isaac.synthetic_utils import shapenet

    parser = argparse.ArgumentParser("Convert ShapeNet assets to USD")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="List of ShapeNet categories to convert (space seperated).",
    )
    parser.add_argument(
        "--max-models", type=int, default=None, help="If specified, convert up to `max-models` per category."
    )
    parser.add_argument(
        "--load-materials", action="store_true", help="If specified, materials will be loaded from shapenet meshes"
    )
    args = parser.parse_args()

    # Ensure Omniverse Kit is launched via OmniKitHelper before shapenet_convert() is called
    shapenet.shapenet_convert(args.categories, args.max_models, args.load_materials)
    # cleanup
    kit.shutdown()
