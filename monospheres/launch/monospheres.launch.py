"""Launch the MonoSpheres node (ov_extension.py).

Arguments:
  config         parameter file name in monospheres/config (default: flightforge.yaml) or an absolute path
  custom_config  optional second parameter file whose values override `config`
  use_sim_time   default true

The pip dependencies live in the package's python-env venv; activate it before launching
(the tmux sessions do this in pre_window).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve(path, cwd_relative):
    if os.path.isabs(path):
        return path
    if cwd_relative:
        return os.path.abspath(path)
    return os.path.join(get_package_share_directory('monospheres'), 'config', path)


def _setup(context):
    config = LaunchConfiguration('config').perform(context)
    custom_config = LaunchConfiguration('custom_config').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() in ('true', '1')

    params = [_resolve(config, cwd_relative=False)]
    if custom_config:
        params.append(_resolve(custom_config, cwd_relative=True))
    params.append({'use_sim_time': use_sim_time})

    return [Node(
        package='monospheres',
        executable='ov_extension.py',
        name='ov_extension_node',
        output='both',  # screen + ~/.ros/log/<run>/ov_extension.py-*-stdout.log
        emulate_tty=True,
        parameters=params,
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='flightforge.yaml'),
        DeclareLaunchArgument('custom_config', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
