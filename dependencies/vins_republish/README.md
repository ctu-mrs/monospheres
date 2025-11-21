
# VINS republisher

Transformes the VIO odometry to the reference frame expected by the [mrs_uav_odometry](https://github.com/ctu-mrs/mrs_uav_odometry).

## Usage

Use one of the launch files in the `launch/` folder, for example:
```
roslaunch vins_republisher vins_republisher.launch
```
or create your own by copying an existing launch file and modifying the transformations.


## External dependencies

ROS1, mrs_lib
