from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_rviz = get_package_share_directory('monospheres') + '/rviz/flightforge.rviz'
    return LaunchDescription([
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
             arguments=['-d', LaunchConfiguration('rviz_config')],
             parameters=[{'use_sim_time': True}]),
    ])
