# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import enum
import random
from collections import deque
import numpy as np
import os
import io
import cv2
import PIL.Image
import pickle
from typing import List, Set, Dict, Tuple, Optional
from .priority_queue import *


DEFAULT_IMAGE_SIZE = (32, 32)


def mask_L(size=256, thickness=1):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size, 0), size // 2, (255, 255, 255), thickness)
    return PIL.Image.fromarray(mask)


def mask_I(size=256, thickness=1):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.line(mask, (size // 2, 0), (size // 2, size), (255, 255, 255), thickness, cv2.LINE_4)
    return PIL.Image.fromarray(mask)


def mask_T(size=256, thickness=1):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask = np.maximum(mask, cv2.circle(mask, (0, size), size // 2, (255, 255, 255), thickness))
    mask = np.maximum(mask, cv2.circle(mask, (size, size), size // 2, (255, 255, 255), thickness))
    mask = np.maximum(mask, cv2.line(mask, (0, size // 2), (size, size // 2), (255, 255, 255), thickness, cv2.LINE_4))
    return PIL.Image.fromarray(mask)


def mask_X(size=256, thickness=1):
    mask = mask_L(size, thickness)
    mask = np.maximum(mask, mask_I(size, thickness))
    for i in range(4):
        mask = np.maximum(mask, cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE))
    return PIL.Image.fromarray(mask)


_I_IMAGE = mask_I()
_L_IMAGE = mask_L()
_T_IMAGE = mask_T()
_X_IMAGE = mask_X()


class RoadBlockType(enum.IntEnum):
    EMPTY = 0
    I = 1
    L = 2
    T = 3
    X = 4

    def ports(self):
        if self == RoadBlockType.I:
            return [0, 1, 0, 1]  # left, top, right, bottom
        elif self == RoadBlockType.L:
            return [0, 1, 1, 0]
        elif self == RoadBlockType.T:
            return [1, 0, 1, 1]
        elif self == RoadBlockType.X:
            return [1, 1, 1, 1]
        else:
            return [0, 0, 0, 0]

    def image(self, size=DEFAULT_IMAGE_SIZE):
        if self == RoadBlockType.I:
            return _I_IMAGE.resize(size)
        elif self == RoadBlockType.L:
            return _L_IMAGE.resize(size)
        elif self == RoadBlockType.T:
            return _T_IMAGE.resize(size)
        elif self == RoadBlockType.X:
            return _X_IMAGE.resize(size)
        else:
            return PIL.Image.fromarray(np.zeros(size + (3,), dtype=np.uint8))

    def paths_mask(self, size=DEFAULT_IMAGE_SIZE, thickness=1):
        if self == RoadBlockType.I:
            return mask_I(size[0], thickness)
        elif self == RoadBlockType.L:
            return mask_L(size[0], thickness)
        elif self == RoadBlockType.T:
            return mask_T(size[0], thickness)
        elif self == RoadBlockType.X:
            return mask_X(size[0], thickness)
        else:
            return PIL.Image.fromarray(np.zeros(size, dtype=np.uint8))


class RoadBlockState(enum.IntEnum):
    HIDDEN = 0
    UP = 1  # 0
    DOWN = 2  # 180
    LEFT = 3  # CCW 90
    RIGHT = 4  # CW  90

    @staticmethod
    def random():
        return RoadBlockState(np.random.randint(len(RoadBlockState)))


class RoadBlock(object):
    def __init__(self, type: RoadBlockType, state: RoadBlockState):
        self.type = type
        self.state = state

    def __iter__(self):
        yield self.type
        yield self.state

    def ports(self):
        if self.state == RoadBlockState.HIDDEN:
            return [0, 0, 0, 0]
        elif self.state == RoadBlockState.UP:
            return self.type.ports()
        elif self.state == RoadBlockState.DOWN:
            return list(np.roll(self.type.ports(), 2))
        elif self.state == RoadBlockState.LEFT:
            return list(np.roll(self.type.ports(), -1))
        else:
            return list(np.roll(self.type.ports(), 1))

    def has_left_port(self):
        return self.ports()[0]

    def has_right_port(self):
        return self.ports()[2]

    def has_top_port(self):
        return self.ports()[1]

    def has_bottom_port(self):
        return self.ports()[3]

    def image(self, size=DEFAULT_IMAGE_SIZE):
        #         if self.state == RoadBlockState.HIDDEN or self.type == RoadBlockType.EMPTY:
        #             return PIL.Image.fromarray(np.zeros(size + (3,), dtype=np.uint8))
        image = self.type.image(size=size)
        if self.state == RoadBlockState.LEFT:
            image = image.rotate(90)
        elif self.state == RoadBlockState.RIGHT:
            image = image.rotate(-90)
        elif self.state == RoadBlockState.DOWN:
            image = image.rotate(180)
        return image

    def paths_mask(self, size=DEFAULT_IMAGE_SIZE, thickness=1):
        #         if self.state == RoadBlockState.HIDDEN or self.type == RoadBlockType.EMPTY:
        #             return PIL.Image.fromarray(np.zeros(size, dtype=np.uint8))
        image = self.type.paths_mask(size=size, thickness=thickness)
        if self.state == RoadBlockState.LEFT:
            image = image.rotate(90)
        elif self.state == RoadBlockState.RIGHT:
            image = image.rotate(-90)
        elif self.state == RoadBlockState.DOWN:
            image = image.rotate(180)
        return image


def l1_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class RoadLocation(object):
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def __iter__(self):
        yield self.i
        yield self.j


class RoadMap(object):
    def __init__(self, grid: List[List[RoadBlock]]):
        self.grid = grid

    @staticmethod
    def create_random_from_types(types: List[RoadBlockType], NI, NJ):
        grid = []
        for i in range(NI):
            row = []
            for j in range(NJ):
                row.append(RoadBlock(RoadBlockType.EMPTY, RoadBlockState.random()))
            grid.append(row)

        # construct positions
        locations = []
        for i in range(NI):
            for j in range(NJ):
                locations.append(RoadLocation(i, j))

        np.random.shuffle(locations)
        locations = locations[0 : len(types)]

        for i, loc in enumerate(locations):
            grid[loc.i][loc.j] = RoadBlock(types[i], RoadBlockState.random())

        return RoadMap(grid)

    @staticmethod
    def create_from_numpy(types, states):
        grid = []
        for i in range(types.shape[0]):
            row = []
            for j in range(types.shape[1]):
                row.append(RoadBlock(RoadBlockType(types[i, j]), RoadBlockState(states[i, j])))
            grid.append(row)
        return RoadMap(grid)

    @property
    def NI(self):
        return len(self.grid)

    @property
    def NJ(self):
        return len(self.grid[0])

    def numpy(self):

        types = []
        states = []

        for i in range(self.NI):

            types_i = []
            states_i = []

            for j in range(self.NJ):

                types_i.append(int(self.grid[i][j].type))
                states_i.append(int(self.grid[i][j].state))

            types.append(types_i)
            states.append(states_i)

        return np.array(types), np.array(states)

    def _children(self, i, j):
        block = self.grid[i][j]
        children = []
        if i > 0:
            top = self.grid[i - 1][j]
            if top.has_bottom_port() and block.has_top_port():
                children.append((i - 1, j))
        if i < self.NI - 1:
            bottom = self.grid[i + 1][j]
            if bottom.has_top_port() and block.has_bottom_port():
                children.append((i + 1, j))
        if j > 0:
            left = self.grid[i][j - 1]
            if left.has_right_port() and block.has_left_port():
                children.append((i, j - 1))
        if j < self.NJ - 1:
            right = self.grid[i][j + 1]
            if right.has_left_port() and block.has_right_port():
                children.append((i, j + 1))
        return children

    def _search_path(self, i, j, visited):

        q = deque()

        q.append((i, j))
        path = []
        while q:

            i, j = q.popleft()
            path.append((i, j))

            for child in self._children(i, j):
                if not visited[child[0], child[1]]:
                    q.append(child)
                    visited[child[0], child[1]] = True

        return path

    def find_shortest_path(self, a, b):

        visited = np.zeros((self.NI, self.NJ), dtype=np.bool)

        q = PriorityQueue()
        q.push((l1_distance(a, b), [a]))
        visited[a[0], a[1]] = 1

        while not q.empty():

            cost, path = q.pop()
            tail = path[-1]

            if tail[0] == b[0] and tail[1] == b[1]:
                return path

            for child in self._children(tail[0], tail[1]):
                if not visited[child[0], child[1]]:
                    child_path = path + [child]
                    child_cost = len(child_path) + l1_distance(child, b)
                    q.push((child_cost, child_path))
                    visited[child[0], child[1]] = 1

        return None

    def paths(self):
        visited = np.zeros((self.NI, self.NJ), dtype=np.bool)

        # set blocks that cannot be path components as visited
        for i in range(self.NI):
            for j in range(self.NJ):
                block = self.grid[i][j]
                if block.state == RoadBlockState.HIDDEN or block.type == RoadBlockType.EMPTY:
                    visited[i, j] = True

        paths = []
        for i in range(self.NI):
            for j in range(self.NJ):
                if not visited[i, j]:
                    visited[i, j] = True
                    path = self._search_path(i, j, visited)
                    paths.append(path)

        return paths

    def num_open_ports(self):
        num_open = 0
        for i in range(self.NJ):
            for j in range(self.NI):
                num_open += np.count_nonzero(self.grid[i][j].ports()) - len(self._children(i, j))
        return num_open

    def num_ports(self):
        num_ports = 0
        for i in range(self.NJ):
            for j in range(self.NI):
                num_ports += np.count_nonzero(self.grid[i][j].ports())  # - len(self._children(i, j))
        return num_ports

    def num_closed_ports(self):
        num_ports = 0
        for i in range(self.NJ):
            for j in range(self.NI):
                num_ports += len(self._children(i, j))
        return num_ports

    def image(self, block_size=DEFAULT_IMAGE_SIZE):
        si = block_size[0]
        sj = block_size[1]
        image = np.zeros((si * self.NI, sj * self.NJ, 3), dtype=np.uint8)
        for i in range(self.NJ):
            for j in range(self.NI):
                image[i * si : i * si + si, j * sj : j * sj + sj] = np.array(self.grid[i][j].image(size=block_size))
        return PIL.Image.fromarray(image)

    def paths_mask(self, block_size=DEFAULT_IMAGE_SIZE, thickness=1):
        si = block_size[0]
        sj = block_size[1]
        image = np.zeros((si * self.NI, sj * self.NJ), dtype=np.uint8)
        for i in range(self.NJ):
            for j in range(self.NI):
                image[i * si : i * si + si, j * sj : j * sj + sj] = np.array(
                    self.grid[i][j].paths_mask(size=block_size, thickness=thickness)
                )
        return PIL.Image.fromarray(image)

    def obs(self):
        obs = np.zeros((4, self.NI, self.NJ), dtype=np.float32)
        for i in range(self.NI):
            for j in range(self.NJ):
                obs[0, i, j] = self.grid[i][j].has_left_port()
                obs[1, i, j] = self.grid[i][j].has_top_port()
                obs[2, i, j] = self.grid[i][j].has_right_port()
                obs[3, i, j] = self.grid[i][j].has_bottom_port()
        return obs

    def swap_(self, a, b):
        tmp = self.grid[a[0]][a[1]]
        self.grid[a[0]][a[1]] = self.grid[b[0]][b[1]]
        self.grid[b[0]][b[1]] = tmp

    def render(self, widget):
        # Render the environment to the screen
        imgByteArr = io.BytesIO()
        self.image(block_size=(64, 64)).save(imgByteArr, format="PNG")
        imgByteArr = imgByteArr.getvalue()
        widget.value = imgByteArr

    def save(self, f):
        types, states = self.numpy()
        data = {"types": types, "states": states}
        if isinstance(f, str):
            with open(f, "wb") as f:
                pickle.dump(data, f)
        else:
            pickle.dump(data, f)

    @staticmethod
    def load(f):
        if isinstance(f, str):
            with open(f, "rb") as f:
                data = pickle.load(f)
        else:
            data = pickle.load(f)
        return RoadMap.create_from_numpy(data["types"], data["states"])

    def ports(self):
        ports = np.zeros((self.NI, self.NJ, 4), np.bool)
        for i in range(self.NI):
            for j in range(self.NJ):
                ports[i, j, 0] = self.grid[i][j].has_left_port()
                ports[i, j, 1] = self.grid[i][j].has_top_port()
                ports[i, j, 2] = self.grid[i][j].has_right_port()
                ports[i, j, 3] = self.grid[i][j].has_bottom_port()
        return ports

    @staticmethod
    def create_from_ports(ports):
        NI = ports.shape[0]
        NJ = ports.shape[1]
        types = np.zeros(ports.shape[0:2], dtype=np.int64)
        states = np.zeros(ports.shape[0:2], dtype=np.int64)
        for i in range(NI):
            for j in range(NJ):
                pij = ports[i, j]
                for typ in RoadBlockType:
                    if (np.roll(typ.ports(), 0) == pij).all():
                        types[i, j] = typ
                        states[i, j] = RoadBlockState.UP
                        break
                    elif (np.roll(typ.ports(), 1) == pij).all():
                        types[i, j] = typ
                        states[i, j] = RoadBlockState.RIGHT
                        break
                    elif (np.roll(typ.ports(), 2) == pij).all():
                        types[i, j] = typ
                        states[i, j] = RoadBlockState.DOWN
                        break
                    elif (np.roll(typ.ports(), 3) == pij).all():
                        types[i, j] = typ
                        states[i, j] = RoadBlockState.LEFT
                        break
        return RoadMap.create_from_numpy(types, states)
