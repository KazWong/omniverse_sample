# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Dataset with online randomized scene generation for Instance Segmentation training.

Use OmniKit to generate a simple scene. At each iteration, the scene is populated by
adding assets from the user-specified classes with randomized pose and colour. 
The camera position is also randomized before capturing groundtruth consisting of
an RGB rendered image, Tight 2D Bounding Boxes and Instance Segmentation masks. 
"""


import os
import glob
import torch
import random
import numpy as np
import signal

import carb
import omni

from omni.isaac.python_app import OmniKitHelper

# Setup default generation variables
# Value are (min, max) ranges
RANDOM_TRANSLATION_X = (-30.0, 30.0)
RANDOM_TRANSLATION_Z = (-30.0, 30.0)
RANDOM_ROTATION_Y = (0.0, 360.0)
SCALE = 20
CAMERA_DISTANCE = 300
BBOX_AREA_THRESH = 16

# Default rendering parameters
RENDER_CONFIG = {
    "renderer": "PathTracing",
    "samples_per_pixel_per_frame": 12,
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
}


class RandomObjects(torch.utils.data.IterableDataset):
    """Dataset of random ShapeNet objects.
    Objects are randomly chosen from selected categories and are positioned, rotated and coloured
    randomly in an empty room. RGB, BoundingBox2DTight and Instance Segmentation are captured by moving a
    camera aimed at the centre of the scene which is positioned at random at a fixed distance from the centre.

    This dataset is intended for use with ShapeNet but will function with any dataset of USD models
    structured as `root/category/**/*.usd. One note is that this is designed for assets without materials
    attached. This is to avoid requiring to compile MDLs and load textures while training.

    Args:
        categories (tuple of str): Tuple or list of categories. For ShapeNet, these will be the synset IDs.
        max_asset_size (int): Maximum asset file size that will be loaded. This prevents out of memory errors
            due to loading large meshes.
        num_assets_min (int): Minimum number of assets populated in the scene.
        num_assets_max (int): Maximum number of assets populated in the scene.
        split (float): Fraction of the USDs found to use for training.
        train (bool): If true, use the first training split and generate infinite random scenes.
    """

    def __init__(
        self, root, categories, max_asset_size=None, num_assets_min=3, num_assets_max=5, split=0.7, train=True
    ):
        assert len(categories) > 1
        assert (split > 0) and (split <= 1.0)

        self.kit = OmniKitHelper(config=RENDER_CONFIG)
        from omni.isaac.synthetic_utils import SyntheticDataHelper, DomainRandomization
        from omni.isaac.synthetic_utils import shapenet

        self.sd_helper = SyntheticDataHelper()
        self.dr_helper = DomainRandomization()
        self.dr_helper.toggle_manual_mode()
        self.stage = self.kit.get_stage()

        from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        self.asset_path = nucleus_server + "/Isaac"

        # If ShapeNet categories are specified with their names, convert to synset ID
        # Remove this if using with a different dataset than ShapeNet
        category_ids = [shapenet.LABEL_TO_SYNSET.get(c, c) for c in categories]
        self.categories = category_ids
        self.range_num_assets = (num_assets_min, max(num_assets_min, num_assets_max))
        self.references = self._find_usd_assets(root, category_ids, max_asset_size, split, train)

        self._setup_world()
        self.cur_idx = 0
        self.exiting = False

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        self.exiting = True

    def _setup_world(self):
        from pxr import UsdGeom

        """Setup lights, walls, floor, ceiling and camera"""
        # In a practical setting, the room parameters should attempt to match those of the
        # target domain. Here, we insteady choose for simplicity.
        self.kit.create_prim(
            "/World/Room", "Sphere", attributes={"radius": 1e3, "primvars:displayColor": [(1.0, 1.0, 1.0)]}
        )
        self.kit.create_prim(
            "/World/Ground",
            "Cylinder",
            translation=(0.0, -0.5, 0.0),
            rotation=(90.0, 0.0, 0.0),
            attributes={"height": 1, "radius": 1e4, "primvars:displayColor": [(1.0, 1.0, 1.0)]},
        )
        self.kit.create_prim(
            "/World/Light1",
            "SphereLight",
            translation=(-450, 350, 350),
            attributes={"radius": 100, "intensity": 30000.0, "color": (0.0, 0.365, 0.848)},
        )
        self.kit.create_prim(
            "/World/Light2",
            "SphereLight",
            translation=(450, 350, 350),
            attributes={"radius": 100, "intensity": 30000.0, "color": (1.0, 0.278, 0.0)},
        )
        self.kit.create_prim("/World/Asset", "Xform")

        self.camera_rig = UsdGeom.Xformable(self.kit.create_prim("/World/CameraRig", "Xform"))
        self.camera = self.kit.create_prim("/World/CameraRig/Camera", "Camera", translation=(0.0, 0.0, CAMERA_DISTANCE))
        vpi = omni.kit.viewport.get_viewport_interface()
        vpi.get_viewport_window().set_active_camera(str(self.camera.GetPath()))
        self.viewport = omni.kit.viewport.get_default_viewport_window()
        self.kit.update()

        # create DR components
        self.create_dr_comp()

        self.kit.update()

    def _find_usd_assets(self, root, categories, max_asset_size, split, train=True):
        """Look for USD files under root/category for each category specified.
        For each category, generate a list of all USD files found and select
        assets up to split * len(num_assets) if `train=True`, otherwise select the
        remainder.
        """
        references = {}
        for category in categories:
            all_assets = glob.glob(os.path.join(root, category, "**/*.usd"), recursive=True)

            # Filter out large files (which can prevent OOM errors during training)
            if max_asset_size is None:
                assets_filtered = all_assets
            else:
                assets_filtered = []
                for a in all_assets:
                    if os.stat(a).st_size > max_asset_size * 1e6:
                        print(f"{a} skipped as it exceeded the max size {max_asset_size} MB.")
                    else:
                        assets_filtered.append(a)

            num_assets = len(assets_filtered)
            if num_assets == 0:
                raise ValueError(f"No USDs found for category {category} under max size {max_asset_size} MB.")

            if train:
                references[category] = assets_filtered[: int(num_assets * split)]
            else:
                references[category] = assets_filtered[int(num_assets * split) :]
        return references

    def load_single_asset(self, ref, semantic_label, suffix=""):
        from pxr import UsdGeom

        """Load a USD asset with random pose.
        args
            ref (str): Path to the USD that this prim will reference.
            semantic_label (str): Semantic label.
            suffix (str): String to add to the end of the prim's path.
        """
        x = random.uniform(*RANDOM_TRANSLATION_X)
        z = random.uniform(*RANDOM_TRANSLATION_Z)
        rot_y = random.uniform(*RANDOM_ROTATION_Y)
        asset = self.kit.create_prim(
            f"/World/Asset/mesh{suffix}",
            "Xform",
            scale=(SCALE, SCALE, SCALE),
            rotation=(0.0, rot_y, 0.0),
            ref=ref,
            semantic_label=semantic_label,
        )
        bound = UsdGeom.Mesh(asset).ComputeWorldBound(0.0, "default")
        box_min_y = bound.GetBox().GetMin()[1]
        UsdGeom.XformCommonAPI(asset).SetTranslate((x, -box_min_y, z))
        return asset

    def populate_scene(self):
        """Clear the scene and populate it with assets."""
        self.stage.RemovePrim("/World/Asset")
        self.assets = []
        num_assets = random.randint(*self.range_num_assets)
        for i in range(num_assets):
            category = random.choice(list(self.references.keys()))
            ref = random.choice(self.references[category])
            self.assets.append(self.load_single_asset(ref, category, i))

    def randomize_camera(self):
        """Randomize the camera position."""
        # By simply rotating a camera "rig" instead repositioning the camera
        # itself, we greatly simplify our job.

        # Clear previous transforms
        self.camera_rig.ClearXformOpOrder()
        # Change azimuth angle
        self.camera_rig.AddRotateYOp().Set(random.random() * 360)
        # Change elevation angle
        self.camera_rig.AddRotateXOp().Set(random.random() * -90)

    def create_dr_comp(self):
        """Creates DR components with various attributes.
        The asset prims to randomize is an empty list for most components
        since we get a new list of assets every iteration.
        The asset list will be updated for each component in update_dr_comp()
        """
        texture_list = [
            self.asset_path + "/Samples/DR/Materials/Textures/checkered.png",
            self.asset_path + "/Samples/DR/Materials/Textures/marble_tile.png",
            self.asset_path + "/Samples/DR/Materials/Textures/picture_a.png",
            self.asset_path + "/Samples/DR/Materials/Textures/picture_b.png",
            self.asset_path + "/Samples/DR/Materials/Textures/textured_wall.png",
            self.asset_path + "/Samples/DR/Materials/Textures/checkered_color.png",
        ]
        material_list = [
            self.asset_path + "/Samples/DR/Materials/checkered.mdl",
            self.asset_path + "/Samples/DR/Materials/checkered_color.mdl",
            self.asset_path + "/Samples/DR/Materials/marble_tile.mdl",
            self.asset_path + "/Samples/DR/Materials/picture_a.mdl",
            self.asset_path + "/Samples/DR/Materials/picture_b.mdl",
            self.asset_path + "/Samples/DR/Materials/textured_wall.mdl",
        ]
        light_list = ["World/Light1", "World/Light2"]
        self.texture_comp = self.dr_helper.create_texture_comp([], True, texture_list)
        self.color_comp = self.dr_helper.create_color_comp([])
        self.material_comp = self.dr_helper.create_material_comp([], material_list)
        self.movement_comp = self.dr_helper.create_movement_comp([])
        self.rotation_comp = self.dr_helper.create_rotation_comp([])
        self.scale_comp = self.dr_helper.create_scale_comp([], max_range=(50, 50, 50))
        self.light_comp = self.dr_helper.create_light_comp(light_list)
        self.visibility_comp = self.dr_helper.create_visibility_comp([])

    def update_dr_comp(self, dr_comp):
        """Updates DR component with the asset prim paths that will be randomized"""
        comp_prim_paths_target = dr_comp.GetPrimPathsRel()
        comp_prim_paths_target.ClearTargets(True)
        for asset in self.assets:
            comp_prim_paths_target.AddTarget(asset.GetPrimPath())

    def __iter__(self):
        return self

    def __next__(self):
        # Generate a new scene
        self.populate_scene()
        self.randomize_camera()

        """The below update calls set the paths of prims that need to be randomized
        with the settings provided in their corresponding DR create component
        """

        # In this example, either update texture or color or material of assets
        # self.update_dr_comp(self.color_comp)
        self.update_dr_comp(self.texture_comp)
        # self.update_dr_comp(self.material_comp)

        # Also update movement, rotation and scale components
        # self.update_dr_comp(self.movement_comp)
        # self.update_dr_comp(self.rotation_comp)
        self.update_dr_comp(self.scale_comp)

        # randomize once
        self.dr_helper.randomize_once()

        # step once and then wait for materials to load
        self.kit.update()
        print("waiting for materials to load...")
        while self.kit.is_loading():
            self.kit.update()
        print("done")
        self.kit.update()
        # Collect Groundtruth
        gt = self.sd_helper.get_groundtruth(["rgb", "boundingBox2DTight", "instanceSegmentation"], self.viewport)

        # RGB
        # Drop alpha channel
        image = gt["rgb"][..., :3]
        # Cast to tensor if numpy array
        if isinstance(gt["rgb"], np.ndarray):
            image = torch.tensor(image, dtype=torch.float, device="cuda")
        # Normalize between 0. and 1. and change order to channel-first.
        image = image.float() / 255.0
        image = image.permute(2, 0, 1)

        # Bounding Box
        gt_bbox = gt["boundingBox2DTight"]

        # Create mapping from categories to index
        mapping = {cat: i + 1 for i, cat in enumerate(self.categories)}
        bboxes = torch.tensor(gt_bbox[["x_min", "y_min", "x_max", "y_max"]].tolist())
        # For each bounding box, map semantic label to label index
        labels = torch.LongTensor([mapping[bb["semanticLabel"]] for bb in gt_bbox])

        # Calculate bounding box area for each area
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # Idenfiy invalid bounding boxes to filter final output
        valid_areas = (areas > 0.0) * (areas < (image.shape[1] * image.shape[2]))

        # Instance Segmentation
        instance_data, instance_mappings = gt["instanceSegmentation"][0], gt["instanceSegmentation"][1]
        instance_list = [im[0] for im in gt_bbox]
        masks = np.zeros((len(instance_list), *instance_data.shape), dtype=np.bool)
        for i, instances in enumerate(instance_list):
            masks[i] = np.isin(instance_data, instances)
        if isinstance(masks, np.ndarray):
            masks = torch.tensor(masks, device="cuda")

        target = {
            "boxes": bboxes[valid_areas],
            "labels": labels[valid_areas],
            "masks": masks[valid_areas],
            "image_id": torch.LongTensor([self.cur_idx]),
            "area": areas[valid_areas],
            "iscrowd": torch.BoolTensor([False] * len(bboxes[valid_areas])),  # Assume no crowds
        }

        self.cur_idx += 1
        return image, target


if __name__ == "__main__":
    "Typical usage"
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser("Dataset test")
    parser.add_argument("--categories", type=str, nargs="+", required=True, help="List of object classes to use")
    parser.add_argument(
        "--max-asset-size",
        type=float,
        default=10.0,
        help="Maximum asset size to use in MB. Larger assets will be skipped.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing USDs. If not specified, use {SHAPENET_LOCAL_DIR}_nomat as root.",
    )
    args = parser.parse_args()

    # If root is not specified use the environment variable SHAPENET_LOCAL_DIR with the _nomat suffix as root
    if args.root is None:
        args.root = f"{os.path.abspath(os.environ['SHAPENET_LOCAL_DIR'])}_mat"

    dataset = RandomObjects(args.root, args.categories, max_asset_size=args.max_asset_size)
    from omni.isaac.synthetic_utils import visualization as vis
    from omni.isaac.synthetic_utils import shapenet

    categories = [shapenet.LABEL_TO_SYNSET.get(c, c) for c in args.categories]

    # Iterate through dataset and visualize the output
    plt.ion()
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.tight_layout()
    for image, target in dataset:
        for ax in axes:
            ax.clear()
            ax.axis("off")

        np_image = image.permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(np_image)

        num_instances = len(target["boxes"])
        colours = vis.random_colours(num_instances)
        overlay = np.zeros_like(np_image)
        for mask, colour in zip(target["masks"].cpu().numpy(), colours):
            overlay[mask, :3] = colour

        axes[1].imshow(overlay)
        mapping = {i + 1: cat for i, cat in enumerate(categories)}
        labels = [shapenet.SYNSET_TO_LABEL[mapping[label.item()]] for label in target["labels"]]
        vis.plot_boxes(ax, target["boxes"].tolist(), labels=labels, colours=colours)

        plt.draw()
        plt.pause(0.01)
        plt.savefig("domain_randomization.png")
        if dataset.exiting:
            break
    # cleanup
    dataset.kit.shutdown()
