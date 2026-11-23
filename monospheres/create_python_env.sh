#!/bin/bash

# get the path to this script
MY_PATH=`dirname "$0"`
MY_PATH=`( cd "$MY_PATH" && pwd )`
cd "$MY_PATH"

# create the environment; system site packages give access to rclpy and the
# ROS 2 message packages, plus the system numpy and cv2 that ROS is built against
python3 -m venv --system-site-packages python-env

# activate the environment
source ./python-env/bin/activate

# install the requirements
python3 -m pip install -r requirements.txt
