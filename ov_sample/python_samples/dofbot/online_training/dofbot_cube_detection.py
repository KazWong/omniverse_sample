# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Dofbot Cube Detection Demonstration

Use a PyTorch dataloader together with OmniKit to generate scenes and groundtruth to
train a [MobileNetV3](https://arxiv.org/abs/1905.02244) model.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import signal

from dofbot_dataset import RandomObjects
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def main(args):
    device = "cuda"

    # Setup data
    train_set = RandomObjects()
    train_loader = DataLoader(train_set, batch_size=2, collate_fn=lambda x: tuple(zip(*x)))

    def handle_exit(self, *args, **kwargs):
        print("exiting cube detection dataset generation...")
        train_set.exiting = True

    signal.signal(signal.SIGINT, handle_exit)

    from omni.isaac.synthetic_utils import visualization as vis

    # Setup Model
    if args.eval_model == "":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=False, num_classes=1 + len(args.categories)
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        model = torch.load(args.eval_model)

    if args.visualize:
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for i, train_batch in enumerate(train_loader):
        if i > args.max_iters or train_set.exiting:
            print("Exiting ...")
            train_set.kit.shutdown()
            break

        if args.eval_model == "":
            model.train()

        images, targets = train_batch
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if args.eval_model == "":
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            print(f"ITER {i} | {loss:.6f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(images[:1])

            if args.visualize:
                idx = 0
                score_thresh = 0.5

                pred = predictions[idx]

                np_image = images[idx].permute(1, 2, 0).cpu().numpy()
                for ax in axes:
                    if args.eval_model == "":
                        fig.suptitle(f"Iteration {i:05} \n {loss:.6f}", fontsize=14)
                    else:
                        fig.suptitle(f"Iteration {i:05} \n Evaluating", fontsize=14)
                    ax.cla()
                    ax.axis("off")
                    ax.imshow(np_image)
                axes[0].set_title("Input")
                axes[1].set_title("Input + Predictions")

                score_filter = [i for i in range(len(pred["scores"])) if pred["scores"][i] > score_thresh]
                num_instances = len(score_filter)
                colours = vis.random_colours(num_instances, enable_random=False)

                mapping = {i + 1: cat for i, cat in enumerate(args.categories)}
                labels = [mapping[label.item()] for label in pred["labels"]]
                vis.plot_boxes(ax, pred["boxes"].tolist(), labels=labels, colours=colours, label_size=10)

                if not labels:
                    axes[1].set_title("None")

                plt.draw()
                plt.savefig("train.png")

                # save every 100 steps
                if i % 100 == 0 and args.eval_model == "":
                    torch.save(model, "cube_model_" + str(i) + ".pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Dataset test")

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-iters", type=float, default=1000, help="Number of training iterations.")
    parser.add_argument("--visualize", action="store_true", help="Visualize predicted bounding boxes during training.")
    parser.add_argument("--eval_model", help="model file to evaluate", default="", type=str)
    args = parser.parse_args()

    # Temporary
    args.visualize = True
    args.categories = ["None", "Cube", "Sphere", "Cone"]
    main(args)
