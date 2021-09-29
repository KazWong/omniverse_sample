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

import omni
from omni.isaac.python_app import OmniKitHelper

# to work around torch's SSL issue
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

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
    "headless": False,
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
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from omni.isaac.synthetic_utils import shapenet

        self.sd_helper = SyntheticDataHelper()
        self.stage = self.kit.get_stage()

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

    def _add_preview_surface(self, prim, diffuse, roughness, metallic):
        from pxr import UsdShade, Sdf

        """Add a preview surface material using the metallic workflow."""
        path = f"{prim.GetPath()}/mat"
        material = UsdShade.Material.Define(self.stage, path)
        pbrShader = UsdShade.Shader.Define(self.stage, f"{path}/shader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(diffuse)
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader, "surface")

        UsdShade.MaterialBindingAPI(prim).Bind(material)

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

    def randomize_asset_material(self):
        """Ranomize asset material properties"""
        for asset in self.assets:
            colour = (random.random(), random.random(), random.random())

            # Here we choose not to have materials unrealistically rough or reflective.
            roughness = random.uniform(0.1, 0.9)

            # Here we choose to have more metallic than non-metallic objects.
            metallic = random.choices([0.0, 1.0], weights=(0.8, 0.2))[0]
            self._add_preview_surface(asset, colour, roughness, metallic)

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

    def __iter__(self):
        return self

    def __next__(self):
        # Generate a new scene
        self.populate_scene()
        self.randomize_camera()
        self.randomize_asset_material()
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
        args.root = f"{os.path.abspath(os.environ['SHAPENET_LOCAL_DIR'])}_nomat"

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
        colours = vis.random_colours(num_instances, enable_random=False)
        overlay = np.zeros_like(np_image)
        for mask, colour in zip(target["masks"].cpu().numpy(), colours):
            overlay[mask, :3] = colour

        axes[1].imshow(overlay)
        mapping = {i + 1: cat for i, cat in enumerate(categories)}
        labels = [shapenet.SYNSET_TO_LABEL[mapping[label.item()]] for label in target["labels"]]
        vis.plot_boxes(ax, target["boxes"].tolist(), labels=labels, colours=colours)

        plt.draw()
        plt.savefig("dataset.png")
        if dataset.exiting:
            break
    # cleanup
    dataset.kit.shutdown()
