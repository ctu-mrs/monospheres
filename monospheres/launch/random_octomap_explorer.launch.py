"""Random-goal octomap exploration baseline (needs mrs_octomap_planner).

Arguments: custom_config (parameter file with exploration_safety_area/*), use_sim_time.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    custom_config = LaunchConfiguration('custom_config').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() in ('true', '1')
    params = []
    if custom_config:
        params.append(os.path.abspath(custom_config))
    params.append({'use_sim_time': use_sim_time})
    return [Node(package='monospheres', executable='random_octomap_explorer.py',
                 name='random_octomap_explorer', output='screen', emulate_tty=True,
                 parameters=params)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('custom_config', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
