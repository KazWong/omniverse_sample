# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from torchvision.transforms import ColorJitter
import PIL
import numpy as np

import carb
import omni
import omni.kit.app

from pxr import UsdGeom, Gf, Sdf, Usd, Semantics

import os
import json
import time
import atexit
import asyncio
import numpy as np
import random
import matplotlib.pyplot as plt
from omni.isaac.synthetic_utils import visualization as vis
from omni.isaac.python_app import OmniKitHelper
from omni.isaac.synthetic_utils import SyntheticDataHelper

from jetracer import Jetracer
from track_environment import Environment
from gtc2020_track_utils import *

import gym
from gym import spaces


class JetracerEnv:
    metadata = {"render.modes": ["human"]}

    # TODO : Extract more training options

    def __init__(
        self,
        omni_kit,
        z_height=0,
        max_resets=10,
        updates_per_step=3,
        steps_per_rollout=500,
        mirror_mode=False,
        backwards_term_mode=0,
        reward_mode=0,
    ):

        self.MIRROR_MODE = mirror_mode
        self.BACKWARDS_TERMINATION_MODE = backwards_term_mode
        self.REWARD_MODE = reward_mode

        print("MIRROR_MODE = {}".format(self.MIRROR_MODE))
        print("BACKWARDS_TERMINATION_MODE = {}".format(self.BACKWARDS_TERMINATION_MODE))
        print("REWARD_MODE = {}".format(self.REWARD_MODE))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 6), dtype=np.uint8)

        self.color_jitter = ColorJitter(0.1, 0.05, 0.05, 0.05)
        self.noise = 0.05

        self.dt = 1 / 30.0
        self.omniverse_kit = omni_kit
        self.sd_helper = SyntheticDataHelper()
        self.roads = Environment(self.omniverse_kit)

        # make environment z up
        self.omniverse_kit.set_up_axis(UsdGeom.Tokens.z)

        # generate roads
        self.shape = [6, 6]
        self.roads.generate_road(self.shape)
        self.roads.generate_lights()

        # randomize once to initialize stage
        # the following two lines must be called prior to Jetracer initialization
        # any DR related setup calls should occur before this point
        self.omniverse_kit.update(1 / 60.0)
        self.roads.dr.randomize_once()

        # spawn robot
        self.jetracer = Jetracer(self.omniverse_kit)
        self.initial_loc = self.roads.get_valid_location()
        self.jetracer.spawn(Gf.Vec3d(self.initial_loc[0], self.initial_loc[1], 5), 0)
        self.prev_pose = [0, 0, 0]
        self.current_pose = [0, 0, 0]

        # switch kit camera to jetracer camera
        self.jetracer.activate_camera()

        # start simulation
        self.omniverse_kit.play()

        # Step simulation so that objects fall to rest
        # wait until all materials are loaded
        frame = 0
        print("simulating physics...")
        while frame < 60 or self.omniverse_kit.is_loading():
            self.omniverse_kit.update(self.dt)
            frame = frame + 1
        print("done after frame: ", frame)

        self.initialized = False
        self.numsteps = 0
        self.numresets = 0
        self.maxresets = 10

        # set this to 1 after around 200k steps to randomnize less
        # self.maxresets = 1

        # Randomly mirror horizontally
        self.update_mirror_mode()

    def update_mirror_mode(self):
        # Mirror if mode is enabled and we randomly sample True
        self.mirror_mode = self.MIRROR_MODE & random.choice([False, True])

    def calculate_reward(self):

        # Current and last positions
        pose = np.array([self.current_pose[0], self.current_pose[1]])
        prev_pose = np.array([self.prev_pose[0], self.prev_pose[1]])

        # Finite difference velocity calculation
        vel = pose - prev_pose
        vel_norm = vel
        vel_magnitude = np.linalg.norm(vel)
        if vel_magnitude > 0.0:
            vel_norm = vel / vel_magnitude

        # Distance from the center of the track
        dist = center_line_dist(pose)
        self.dist = dist

        # racing_forward = is_racing_forward(prev_pose, pose)
        # reward = racing_forward * self.current_speed * np.exp(-dist ** 2 / 0.05 ** 2)

        fwd_dir = closest_point_track_direction(pose)
        fwd_dot = np.dot(fwd_dir, vel_norm)

        if self.REWARD_MODE == 0:
            reward = fwd_dot * self.current_speed * np.exp(-dist ** 2 / 0.05 ** 2)
        elif self.REWARD_MODE == 1:
            reward = fwd_dot * self.current_speed

        return reward

    def is_dead(self):
        return not is_outside_track_boundary(np.array([self.current_pose[0], self.current_pose[1]]))

    def transform_action(self, action):

        # If mirrored, swap steering direction
        if self.mirror_mode:
            action[1] = -action[1]

        return action

    def transform_state_image(self, im):

        # If enabled, mirror image horizontally
        if self.mirror_mode:
            return np.flip(im, axis=1)

        return im

    def reset(self):

        # Randomly mirror horizontally
        self.update_mirror_mode()

        if self.numresets % self.maxresets == 0:
            self.roads.reset(self.shape)

        if not self.initialized:
            state, reward, done, info, = self.step([0, 0])
            self.initialized = True

        # Random track point in cm, with a 10 cm stddev gaussian offset
        loc = random_track_point()
        loc = loc + np.random.normal([0.0, 0.0], 10.0)

        # Forward direction at that point
        fwd = closest_point_track_direction(loc)

        # Forward angle in degrees, with a 10 degree stddev gaussian offset
        rot = np.arctan2(fwd[1], fwd[0])
        rot = rot * 180.0 / np.pi
        rot = rot + np.random.normal(10.0)

        self.jetracer.teleport(Gf.Vec3d(loc[0], loc[1], 5), rot, settle=True)

        obs = self.jetracer.observations()
        self.current_pose = obs["pose"]
        self.current_speed = np.linalg.norm(np.array(obs["linear_velocity"]))
        self.current_forward_velocity = obs["local_linear_velocity"][0]

        if self.numresets % self.maxresets == 0:
            frame = 0
            while self.omniverse_kit.is_loading():  # or frame < 750:
                self.omniverse_kit.update(self.dt)
                frame += 1

        viewport = omni.kit.viewport.get_default_viewport_window()
        gt = self.sd_helper.get_groundtruth(["rgb"], viewport)
        currentState = gt["rgb"][:, :, :3]
        currentState = self.transform_state_image(currentState)

        img = np.concatenate((currentState, currentState), axis=2)
        img = np.clip((255 * self.noise * np.random.randn(224, 224, 6) + img.astype(np.float)), 0, 255).astype(np.uint8)

        self.numsteps = 0
        self.previousState = currentState
        self.numresets += 1

        return img

    def is_driving_backwards(self):
        # TODO : Refactor, the bulk of this code is shared with the reward function.
        #        Also, find out at what point in an iteration this is called,
        #        compared to the reward, physics and stuff.
        #        If off by a timestep it's close enough, probably won't cause any issues.

        # Current and last positions
        pose = np.array([self.current_pose[0], self.current_pose[1]])
        prev_pose = np.array([self.prev_pose[0], self.prev_pose[1]])

        # Finite difference velocity calculation
        vel = pose - prev_pose
        vel_norm = vel
        vel_magnitude = np.linalg.norm(vel)
        if vel_magnitude > 0.0:
            vel_norm = vel / vel_magnitude

        # Forward direction on the track
        fwd_dir = closest_point_track_direction(pose)

        # Normalized velocity projected onto the forward direction
        fwd_dot = np.dot(fwd_dir, vel_norm)

        # Going backwards more than 3*pi/8 radians
        return fwd_dot < np.cos(7.0 * np.pi / 8.0)

    def step(self, action):
        print("Number of steps ", self.numsteps)

        # print("Action ", action)

        transformed_action = self.transform_action(action)
        self.jetracer.command(transformed_action)
        frame = 0
        total_reward = 0
        reward = 0
        while frame < 3:
            self.omniverse_kit.update(self.dt)
            obs = self.jetracer.observations()
            self.prev_pose = self.current_pose
            self.current_pose = obs["pose"]
            self.current_speed = np.linalg.norm(np.array(obs["linear_velocity"]))
            self.current_forward_velocity = obs["local_linear_velocity"][0]

            reward = self.calculate_reward()
            done = self.is_dead()

            total_reward += reward
            frame = frame + 1

        viewport = omni.kit.viewport.get_default_viewport_window()
        gt = self.sd_helper.get_groundtruth(["rgb"], viewport)

        currentState = gt["rgb"][:, :, :3]
        currentState = self.transform_state_image(currentState)

        if not self.initialized:
            self.previousState = currentState

        img = np.concatenate((currentState, self.previousState), axis=2)
        img = np.clip((255 * self.noise * np.random.randn(224, 224, 6) + img.astype(np.float)), 0, 255).astype(np.uint8)

        self.previousState = currentState

        other = np.array(
            [*obs["pose"], *obs["linear_velocity"], *obs["local_linear_velocity"], *obs["angular_velocity"]]
        )
        other = np.expand_dims(other.astype(float), 0)

        self.numsteps += 1
        if done:
            print("robot is dead")

        if self.numsteps > 500:
            done = True
            print("robot stepped 500 times")

        if self.dist > LANE_WIDTH:
            print("robot out of bounds. dist = ", self.dist)
            done = True

        if self.BACKWARDS_TERMINATION_MODE == 0:
            if self.current_forward_velocity <= -35:
                print("robot was going backwards forward velocity = ", self.current_forward_velocity)
                done = True

        elif self.BACKWARDS_TERMINATION_MODE == 1:
            if self.is_driving_backwards():
                print("Robot was driving backwards")
                done = True

        return img, reward, done, {}
