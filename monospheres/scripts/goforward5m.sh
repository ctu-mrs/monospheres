#!/bin/bash
# Short initial maneuver: fly 4 m forward in the body frame (deeper into the free space the camera sees,
# and gives OpenVINS motion/parallax). The ROS 1 Gazebo sessions used goto_relative [0, 4, 0, 0] with
# the drone spawned facing +y, which is the same motion.
ros2 service call /$UAV_NAME/control_manager/goto_fcu mrs_msgs/srv/Vec4 "{goal: [4.0, 0.0, 0.0, 0.0]}"
