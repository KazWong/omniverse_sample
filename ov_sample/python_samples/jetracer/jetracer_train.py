# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import os
import sys
import json
import signal
import argparse
from argparse import Namespace

from omni.isaac.python_app import OmniKitHelper


from jetracer_model import CustomCNN

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


def train(args):
    CUSTOM_CONFIG = {
        "width": 224,
        "height": 224,
        "renderer": "RayTracedLighting",
        "headless": args.headless,
        "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    }
    omniverse_kit = OmniKitHelper(CUSTOM_CONFIG)

    from jetracer_env import JetracerEnv

    # we disable all anti aliasing in the render because we want to train on the raw camera image.
    omniverse_kit.set_setting("/rtx/post/aa/op", 0)

    env = JetracerEnv(
        omniverse_kit,
        mirror_mode=args.mirror_mode,
        backwards_term_mode=args.backwards_termination_mode,
        reward_mode=args.reward_mode,
        max_resets=args.rand_freq,
    )

    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path="./params/", name_prefix="rl_model")

    net_arch = [512, 256, dict(pi=[128, 64, 32], vf=[128, 64, 32])]
    policy_kwargs = {"net_arch": net_arch, "features_extractor_class": CustomCNN, "activation_fn": torch.nn.ReLU}

    # create a new model
    if args.loaded_checkpoint == "":

        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=args.tensorboard_dir,
            policy_kwargs=policy_kwargs,
            device="cuda",
            n_steps=args.step_freq,
        )
    else:

        model = PPO.load(args.loaded_checkpoint, env)
    model.learn(
        total_timesteps=args.total_steps,
        callback=checkpoint_callback,
        eval_env=env,
        eval_freq=args.eval_freq,
        eval_log_path=args.evaluation_dir,
        reset_num_timesteps=args.reset_num_timesteps,
    )

    model.save(args.checkpoint_name)


def runEval(args):
    CUSTOM_CONFIG = {
        "width": 224,
        "height": 224,
        "renderer": "RayTracedLighting",
        "headless": args.headless,
        "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    }
    # load a zip file to evaluate here. PPO also saves the best model so far in the eval_log folder.
    # You can evaluate those zip files in the params folder as well (i.e params/rl_model_125999_steps.zip)
    agent = PPO.load(args.evaluation_dir + "/best_model.zip", device="cuda")

    omniverse_kit = OmniKitHelper(CUSTOM_CONFIG)

    from jetracer_env import JetracerEnv

    # we disable all anti aliasing in the render because we want to train on the raw camera image.
    omniverse_kit.set_setting("/rtx/post/aa/op", 0)

    env = JetracerEnv(
        omniverse_kit,
        mirror_mode=args.mirror_mode,
        backwards_term_mode=args.backwards_termination_mode,
        reward_mode=args.reward_mode,
        max_resets=args.rand_freq,
    )
    obs = env.reset()

    while True:
        action = agent.predict(obs)
        print(action)
        obs, rew, done, infos = env.step(action[0])
        if done:
            obs = env.reset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--loaded_checkpoint", help="path to checkpoint to be loaded", default="", nargs="?", type=str)

    parser.add_argument("-E", "--eval", help="evaluate checkpoint", action="store_true")

    parser.add_argument(
        "-R", "--reset_num_timesteps", help="reset the current timestep number (used in logging)", action="store_true"
    )

    parser.add_argument(
        "-M", "--mirror_mode", help="reflect images and actions horizontally during training", action="store_true"
    )

    parser.add_argument("-H", "--headless", help="run in headless mode (no GUI)", action="store_true")

    parser.add_argument(
        "--checkpoint_name", help="name of checkpoint file (no suffix)", default="checkpoint_25k", type=str
    )

    parser.add_argument("--tensorboard_dir", help="path to tensorboard log directory", default="tensorboard", type=str)

    parser.add_argument("--evaluation_dir", help="path to evaluation log directory", default="eval_log", type=str)

    parser.add_argument("--save_freq", help="number of steps before saving a checkpoint", default=1000, type=int)

    parser.add_argument("--eval_freq", help="number of steps before running an evaluation", default=1000, type=int)

    parser.add_argument("--step_freq", help="number of steps before executing a PPO update", default=1000, type=int)

    parser.add_argument(
        "--rand_freq", help="number of environment resets before domain randomization", default=10, type=int
    )

    parser.add_argument(
        "--total_steps",
        help="the total number of steps before exiting and saving a final checkpoint",
        default=25000,
        type=int,
    )

    parser.add_argument("--backwards_termination_mode", help="???", default=0, type=int)

    parser.add_argument("--reward_mode", help="???", default=0, type=int)

    parser.add_argument(
        "--experimentFile", help="specify configuration via JSON.  Overrides commandline", default="", type=str
    )

    args = parser.parse_args()

    if args.experimentFile != "":
        args_dict = vars(args)
        if os.path.exists(args.experimentFile):
            with open(args.experimentFile) as f:
                json_args_dict = json.load(f)

                args_dict.update(json_args_dict)
                args = Namespace(**args_dict)

    print("running with args: ", args)

    def handle_exit(*args, **kwargs):
        print("Exiting training...")
        quit()

    signal.signal(signal.SIGINT, handle_exit)

    if args.eval:
        runEval(args)
    else:
        train(args)
