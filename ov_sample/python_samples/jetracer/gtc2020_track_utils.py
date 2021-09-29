# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
from PIL import Image

# TODO : This is custom, specific to the GTC2020 Jetracer course.
#        Make a more general solution.


def line_seg_closest_point(v0, v1, p0):
    # Project p0 onto (v0, v1) line, then clamp to line segment
    d = v1 - v0
    q = p0 - v0

    t = np.dot(q, d) / np.dot(d, d)
    t = np.clip(t, 0, 1)

    return v0 + t * d


def line_seg_distance(v0, v1, p0):
    p = line_seg_closest_point(v0, v1, p0)

    return np.linalg.norm(p0 - p)


# Canonical arc is centered at origin, and goes from 0 to a0 radians
def canonical_arc_distance(R, a0, x):
    a = np.arctan2(x[1], x[0])

    if a < 0:
        a = a + 2 * np.pi

    if a > a0:
        if a < a0 / 2 + np.pi:
            a = a0
        else:
            a = 0

    p = R * np.array([np.cos(a), np.sin(a)])

    return np.linalg.norm(x - p)


def arc_distance(c, r, a0, a1, x):
    # Point relative to arc origin
    x0 = x - c

    # Rotate point to canonical angle (where arc starts at 0)
    c = np.cos(-a0)
    s = np.sin(-a0)
    R = np.array([[c, -s], [s, c]])

    x0 = np.dot(R, x0)

    return canonical_arc_distance(r, a1 - a0, x0)


def closest_point_arc(c, r, a0, a1, x):

    # Direction to point
    x0 = x - c
    x0 = x0 / np.linalg.norm(x0)

    # print(c, x0, r, c + x0 * r)

    return c + x0 * r


# The forward direction at the closest point on an arc
def closest_point_arc_direction(c, r, a0, a1, x):

    # Direction to point
    x0 = x - c
    x0 = x0 / np.linalg.norm(x0)

    # The tangent is unit circle point rotated pi/2 radians
    return np.array([-x0[1], x0[0]])


def arc_endpoints(c, r, a0, a1):

    c0 = np.cos(a0)
    s0 = np.sin(a0)
    c1 = np.cos(a1)
    s1 = np.sin(a1)

    return c + r * np.array([[c0, s0], [c1, s1]])


# Measurements (in meters)
m0 = 7.620
m1 = 10.668
m2 = 5.491
m3 = 3.048
m4 = 4.348
m5 = 5.380

# Track width
w = 1.22
w_2 = w / 2


# Arc arrays
arc_center = np.zeros((4, 2))
arc_radius = np.zeros(4)
arc_angles = np.zeros((4, 2))

# Arcs
# Bottom left
arc_center[0] = np.array([w, w])
arc_radius[0] = w_2
arc_angles[0] = [np.pi, np.pi * 1.5]

# Top left
arc_center[1] = np.array([m3, m0])
arc_radius[1] = m3 - w_2
arc_angles[1] = [1.75 * np.pi, 3 * np.pi]
ep1 = arc_endpoints(arc_center[1], arc_radius[1], arc_angles[1][0], arc_angles[1][1])

# Others
arc_center[2] = np.array([m5, m4])
arc_radius[2] = 0.5 * (2.134 + 0.914)
arc_angles[2] = [0.75 * np.pi, 1.25 * np.pi]
ep2 = arc_endpoints(arc_center[2], arc_radius[2], arc_angles[2][0], arc_angles[2][1])

arc_center[3] = np.array([m2, w])
arc_radius[3] = w_2
arc_angles[3] = [np.pi * 1.5, np.pi * 2.25]
ep3 = arc_endpoints(arc_center[3], arc_radius[3], arc_angles[3][0], arc_angles[3][1])


# line segment points
line_verts = [
    np.array([w_2, w]),
    np.array([w_2, m0]),
    ep1[0],
    ep2[0],
    ep2[1],
    ep3[1],
    np.array([m2, w_2]),
    np.array([w, w_2]),
]


def random_track_point():

    # TODO : Refactor these dimensions, which show up in multiple places
    p = np.random.random(2) * [6.711, 10.668]

    result = track_segment_closest_point(p)

    return result * 100  # convert to cm. TODO standardize all entry points to cm


# Minimum distances to all segments of the track
def track_segment_distance(p):

    d = np.zeros(8)

    d[0] = line_seg_distance(line_verts[0], line_verts[1], p)
    d[1] = line_seg_distance(line_verts[2], line_verts[3], p)
    d[2] = line_seg_distance(line_verts[4], line_verts[5], p)
    d[3] = line_seg_distance(line_verts[6], line_verts[7], p)
    d[4] = arc_distance(arc_center[0], arc_radius[0], arc_angles[0][0], arc_angles[0][1], p)
    d[5] = arc_distance(arc_center[1], arc_radius[1], arc_angles[1][0], arc_angles[1][1], p)
    d[6] = arc_distance(arc_center[2], arc_radius[2], arc_angles[2][0], arc_angles[2][1], p)
    d[7] = arc_distance(arc_center[3], arc_radius[3], arc_angles[3][0], arc_angles[3][1], p)

    return d


def track_segment_closest_point(p):

    d = track_segment_distance(p)

    # If a line segment is the closest
    if np.min(d[:4]) < np.min(d[4:]):
        idx = np.argmin(d[:4], axis=0)

        return line_seg_closest_point(line_verts[idx * 2], line_verts[idx * 2 + 1], p)

    # If an arc is the closest
    else:
        idx = np.argmin(d[4:], axis=0)

        return closest_point_arc(arc_center[idx], arc_radius[idx], arc_angles[idx][0], arc_angles[idx][1], p)


# Distance to closest point on the track
def center_line_dist(p):
    p = 0.01 * p  # Convert from m to cm

    return np.min(track_segment_distance(p))


# Forward vector at the closest point on the center line
def closest_point_track_direction(p):
    p = 0.01 * p  # Convert from m to cm

    d = track_segment_distance(p)

    # If a line segment is the closest
    if np.min(d[:4]) < np.min(d[4:]):
        idx = np.argmin(d[:4], axis=0)

        v = line_verts[idx * 2 + 1] - line_verts[idx * 2]
        return v / np.linalg.norm(v)

    # If an arc is the closest
    else:
        idx = np.argmin(d[4:], axis=0)

        v = closest_point_arc_direction(arc_center[idx], arc_radius[idx], arc_angles[idx][0], arc_angles[idx][1], p)

        # TODO : All arcs are defined counter-clockwise,
        #        but this doesn't always represent the forward direction on the track.
        #        This is a hack to correct the tangent vector on all but one of the arcs.
        if idx != 2:
            v = -v

        return v


LANE_WIDTH = 0.7  # width of whole track is w = 1.22. To get out of bound is > 1.22/2, so around 0.7
TRACK_DIMS = [671, 1066]  # the track is within (0, 0) to (671.1 cm, 1066.8 cm)


def is_racing_forward(prev_pose, curr_pose):
    prev_pose = 0.01 * prev_pose
    curr_pose = 0.01 * curr_pose

    bottom_left_corner = np.array([0, 0])
    top_left_corner = np.array([0, 10.668])
    top_right_corner = np.array([6.711, 10.668])
    bottom_right_corner = np.array([6.711, 0])

    d0 = line_seg_distance(bottom_left_corner, top_left_corner, curr_pose)
    d1 = line_seg_distance(top_left_corner, top_right_corner, curr_pose)
    d2 = line_seg_distance(top_right_corner, bottom_right_corner, curr_pose)
    d3 = line_seg_distance(bottom_right_corner, bottom_left_corner, curr_pose)

    min_d = np.min([d0, d1, d2, d3])

    which_side = np.array([0, 0])
    if min_d == d0:
        which_side = top_left_corner - bottom_left_corner
    elif min_d == d1:
        which_side = top_right_corner - top_left_corner
    elif min_d == d2:
        which_side = bottom_right_corner - top_right_corner
    elif min_d == d3:
        which_side = bottom_left_corner - bottom_right_corner

    which_size_unit = which_side / np.linalg.norm(which_side)

    curr_vel = curr_pose - prev_pose
    curr_vel_norm = np.linalg.norm(curr_vel)

    curr_vel_unit = np.array([0, 0])
    # checking divide by zero
    if curr_vel_norm:
        curr_vel_unit = curr_vel / curr_vel_norm

    return np.dot(curr_vel_unit, which_size_unit)


def is_outside_track_boundary(curr_pose):
    dist = center_line_dist(curr_pose)
    return dist < LANE_WIDTH


if __name__ == "__main__":

    print("Generating test PNGs")

    # scale
    s = 0.02

    H = int(10668 * s)
    W = int(6711 * s)

    d = np.zeros((H, W))
    fwd = np.zeros((H, W, 3))

    h = np.zeros((H, W))

    print(H, W)

    for _ in range(10000):

        p_scaled = np.random.random(2) * [W, H]
        p_meters = p_scaled / s / 1000.0

        # p_proj = line_seg_closest_point(line_verts[6], line_verts[7], p_meters)
        p_proj = track_segment_closest_point(p_meters)
        # print(h.shape, p_scaled, p_meters, p_proj, p_proj * s)
        p_proj = p_proj + np.random.normal([0, 0], 0.1)

        idx = p_proj * s * 1000.0
        idx = np.floor(idx)
        idx = np.clip(idx, [0, 0], [W - 1, H - 1])  # HACK
        idx = idx.astype("int")

        h[idx[1], idx[0]] = h[idx[1], idx[0]] + 1

    for i in range(H):
        y = ((i + 0.5) / s) / 10.0
        if i % 10 == 0:
            print("{:0.1f}%".format(i / H * 100))
        for j in range(W):
            x = ((j + 0.5) / s) / 10.0

            p = np.array([x, y])

            d[i, j] = center_line_dist(p)
            f = closest_point_track_direction(p)
            fwd[i, j] = np.array([0.5 * (f[0] + 1), 0.5 * (f[1] + 1), 0])

    print("100.0%")

    # Images have zero at the top, so we flip vertically
    d = np.flipud(d)
    fwd = np.flip(fwd, axis=0)
    h = np.flipud(h)

    # Distance function
    im = Image.fromarray((d * 255 / np.max(d)).astype("uint8"))
    im.save("dist.png")

    # Track forward vector
    im = Image.fromarray((fwd * 255).astype("uint8"), "RGB")
    im.save("fwd.png")

    # Track forward vector X
    im = Image.fromarray((fwd[:, :, 0] * 255).astype("uint8"))
    im.save("fwd_x.png")

    # Track forward vector Y
    im = Image.fromarray((fwd[:, :, 1] * 255).astype("uint8"))
    im.save("fwd_y.png")

    # H
    h = h / np.max(h)
    im = Image.fromarray((h * 255).astype("uint8"))
    im.save("h.png")
