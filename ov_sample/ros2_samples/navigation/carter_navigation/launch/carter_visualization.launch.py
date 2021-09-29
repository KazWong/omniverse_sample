## Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

## This launch file is only used for seperately running Rviz2 with the carter_navigation configuration.


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    rviz_config_dir = os.path.join(get_package_share_directory("carter_navigation"), "rviz2", "carter_navigation.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="True", description="Flag to enable use_sim_time"),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
                arguments=["-d", rviz_config_dir],
            ),
        ]
    )
