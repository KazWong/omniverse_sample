# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .road_map import *
from .priority_queue import *


def children(occupancy, point):

    NI = occupancy.shape[0]
    NJ = occupancy.shape[1]

    children = []

    if point[0] > 0:
        pt = [point[0] - 1, point[1]]
        children.append(pt)
    if point[0] < NI - 1:
        pt = [point[0] + 1, point[1]]
        children.append(pt)
    if point[1] > 0:
        pt = [point[0], point[1] - 1]
        children.append(pt)
    if point[1] < NJ - 1:
        pt = [point[0], point[1] + 1]
        children.append(pt)

    return children


def l1_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_path(occupancy, point_a, point_b):

    visited = np.copy(occupancy)
    visited[point_a[0], point_b[0]] = 1

    q = PriorityQueue()  # cost heuristic, path...

    for child in children(visited, point_a):
        if not visited[child[0], child[1]]:
            q.push((1 + l1_distance(child, point_b), [child]))
            visited[child[0], child[1]] = 1

    while not q.empty():

        cost, path = q.pop()
        tail = path[-1]

        for child in children(visited, tail):
            if child[0] == point_b[0] and child[1] == point_b[1]:
                return path
            elif not visited[child[0], child[1]]:
                child_cost = len(path) + l1_distance(child, point_b)
                child_path = path + [child]
                q.push((child_cost, child_path))
                visited[child[0], child[1]] = 1

    return None


def add_port(ports, a, b):
    # port order: left,top,right,bottom
    if b[1] > a[1]:
        # b to right of a
        ports[a[0], a[1], 2] = 1
        ports[b[0], b[1], 0] = 1
    elif b[1] < a[1]:
        # b to left of a
        ports[a[0], a[1], 0] = 1
        ports[b[0], b[1], 2] = 1
    elif b[0] > a[0]:
        # b above a
        ports[a[0], a[1], 3] = 1
        ports[b[0], b[1], 1] = 1
    elif b[0] < a[0]:
        # b below a
        ports[a[0], a[1], 1] = 1
        ports[b[0], b[1], 3] = 1


def ports_to_types_states(ports):
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
    return types, states


class RoadMapGenerator(object):
    def generate(self, shape):
        raise NotImplementedError


class LoopRoadMapGenerator(RoadMapGenerator):
    def generate(self, shape):

        GRID_SIZE = shape

        ports = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 4), np.bool)
        occupancy = np.zeros(GRID_SIZE, np.uint8)
        start = (np.random.randint(GRID_SIZE[0]), np.random.randint(GRID_SIZE[1]))

        path = []
        path.append(start)
        occupancy[start[0], start[1]] = 1

        runner = start
        while True:

            # get valid children
            valid_children = []

            for child in children(occupancy, runner):
                if not occupancy[child[0], child[1]]:
                    child_occupancy = np.copy(occupancy)
                    child_occupancy[child[0], child[1]] = 1
                    child_path = find_path(child_occupancy, child, start)
                    if child_path is not None:
                        valid_children.append(child)

            # exit if no valid child paths
            if len(valid_children) == 0:
                break

            # navigate to random child
            idx = np.random.randint(len(valid_children))
            runner = valid_children[idx]
            path.append(runner)
            occupancy[runner[0], runner[1]] = 1

        path = path + find_path(occupancy, runner, start) + [start]

        for i in range(len(path) - 1):
            add_port(ports, path[i], path[i + 1])

        types, states = ports_to_types_states(ports)
        road_map = RoadMap.create_from_numpy(types, states)

        return types, states, road_map
