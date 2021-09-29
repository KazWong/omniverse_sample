# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .road_map import *
from scipy.spatial import KDTree
import cv2
import matplotlib.pyplot as plt
import numpy as np


class RoadMapPathHelper(object):
    def __init__(self, road_map, block_resolution=128, path_thickness_ratio=19 / 32):
        self._road_map = road_map
        self._block_resolution = block_resolution
        self._map_path_mask = np.array(self._road_map.paths_mask((block_resolution, block_resolution), thickness=1))
        self._map_boundary_mask = np.array(
            self._road_map.paths_mask(
                (block_resolution, block_resolution), thickness=int(block_resolution * path_thickness_ratio)
            )
        )
        mask_pts = np.transpose(np.nonzero(self._map_path_mask))
        mask_pts = mask_pts / block_resolution  # get points in grid coordinates
        self._path_kdtree = KDTree(mask_pts)

        boundary_points = np.transpose(np.nonzero(cv2.Laplacian(self._map_boundary_mask, cv2.CV_32F)))
        boundary_points = boundary_points / block_resolution
        self._boundary_kdtree = KDTree(boundary_points)

        # print("boundary points shape! ", boundary_points.shape)
        # plt.imshow(self._map_boundary_mask)
        # plt.show()

    def get_k_nearest_path_points(self, points, k=1):
        dists, indices = self._path_kdtree.query(points, k=k)
        return dists, self._path_kdtree.data[indices]

    def distance_to_path(self, point):
        dists, pts = self.get_k_nearest_path_points(np.array([point]))
        return (float(dists[0]), pts[0])

    def get_k_nearest_boundary_points(self, points, k=1):
        dists, indices = self._boundary_kdtree.query(points, k=k)
        return dists, self._boundary_kdtree.data[indices]

    def distance_to_boundary(self, point):
        dists, pts = self.get_k_nearest_boundary_points(np.array([point]))
        return float(dists[0])

    def is_inside_path_boundary(self, point):
        return (
            self._map_boundary_mask[int(point[0] * self._block_resolution), int(point[1] * self._block_resolution)] > 0
        )
