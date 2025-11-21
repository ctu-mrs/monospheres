#!/bin/bash

path="/home/\$(optenv USER mrs)/bag_files/monoexplo_sim/"

# By default, we record everything.
# Except for this list of EXCLUDED topics:
exclude=(

# IN GENERAL, DON'T RECORD CAMERAS
#
# If you want to record cameras, create a copy of this script
# and place it at your tmux session.
#
# Please, seek an advice of a senior researcher of MRS about
# what can be recorded. Recording too much data can lead to
# ROS communication hiccups, which can lead to eland, failsafe
# or just a CRASH.

'/gazebo/link_states'
'/gazebo/model_states'
# '(.*)/ov_msckf/loop_depth_colored/compressed'
# '(.*)/polygon_enhanced_debug'
'(.*)/polygon_debug'
'(.*)/vio/camera/image_raw/compressed'

'/ov_msckf/loop_depth'
'/ov_msckf/loop_depth/compressed'
'/ov_msckf/loop_depth/compressed/parameter_descriptions'
'/ov_msckf/loop_depth/compressed/parameter_updates'
'/ov_msckf/loop_depth/compressedDepth'
'/ov_msckf/loop_depth/compressedDepth/parameter_descriptions'
'/ov_msckf/loop_depth/compressedDepth/parameter_updates'
'/ov_msckf/loop_depth/theora'
'/ov_msckf/loop_depth/theora/parameter_descriptions'
'/ov_msckf/loop_depth/theora/parameter_updates'
'/ov_msckf/loop_depth_colored'
'/ov_msckf/loop_depth_colored/compressed'
'/ov_msckf/loop_depth_colored/compressed/parameter_descriptions'
'/ov_msckf/loop_depth_colored/compressed/parameter_updates'
'/ov_msckf/loop_depth_colored/compressedDepth'
'/ov_msckf/loop_depth_colored/compressedDepth/parameter_descriptions'
'/ov_msckf/loop_depth_colored/compressedDepth/parameter_updates'
'/ov_msckf/loop_depth_colored/theora'
'/ov_msckf/loop_depth_colored/theora/parameter_descriptions'
'/ov_msckf/loop_depth_colored/theora/parameter_updates'
'/ov_msckf/loop_extrinsic'
'/ov_msckf/loop_feats'
'/ov_msckf/loop_intrinsics'
'/ov_msckf/loop_pose'
'/ov_msckf/odomimu'
'/ov_msckf/pathgt'
'/ov_msckf/pathimu'
'/ov_msckf/points_aruco'
'/ov_msckf/points_msckf'
'/ov_msckf/points_sim'
'/ov_msckf/points_slam'
'/ov_msckf/points_slam/transformed'
'/ov_msckf/posegt'
'/ov_msckf/poseimu'
'/ov_msckf/trackhist'
# '/ov_msckf/trackhist/compressed'
'/ov_msckf/trackhist/compressed/parameter_descriptions'
'/ov_msckf/trackhist/compressed/parameter_updates'
'/ov_msckf/trackhist/compressedDepth'
'/ov_msckf/trackhist/compressedDepth/parameter_descriptions'
'/ov_msckf/trackhist/compressedDepth/parameter_updates'
'/ov_msckf/trackhist/mouse_click'
'/ov_msckf/trackhist/theora'
'/ov_msckf/trackhist/theora/parameter_descriptions'
'/ov_msckf/trackhist/theora/parameter_updates'
# Every topic containing "compressed"
# '(.*)compressed(.*)'
# Every topic containing "image_raw"
'(.*)image_raw'
# Every topic containing "theora"
'(.*)theora(.*)'
# Every topic containing "h264"
# '(.*)h264(.*)'

# '(.*)octomap(.*)'
'(.*)octomap_global_vis/free_cells_vis_array'
'(.*)octomap_global_vis/free_cells_vis_array_throttled'
'(.*)octomap_global_vis/occupied_cells_vis_array'
# '(.*)octomap_global_vis/occupied_cells_vis_array_throttled'
'(.*)octomap_global_vis/octomap_free_centers'
'(.*)octomap_global_vis/octomap_point_cloud_centers'
'(.*)octomap_local_vis/free_cells_vis_array'
# '(.*)octomap_local_vis/free_cells_vis_array_throttled'
'(.*)octomap_local_vis/occupied_cells_vis_array'
'(.*)octomap_local_vis/occupied_cells_vis_array_throttled'
'(.*)octomap_local_vis/octomap_free_centers'
'(.*)octomap_local_vis/octomap_point_cloud_centers'
'(.*)octomap_planner/diagnostics'
'(.*)octomap_planner/visualize_input'
'(.*)octomap_planner/visualize_planner'
'(.*)octomap_planner/visualize_processed'
'(.*)octomap_server/octomap_global_binary'
'(.*)octomap_server/octomap_global_full'
'(.*)octomap_server/octomap_local_binary'
'(.*)octomap_server/octomap_local_full'
# '(.*)ov_msckf(.*)'
'(.*)mavros(.*)'
'(.*)control_manager(.*)'
'(.*)mavlink(.*)'
'(.*)hw_api(.*)'
'(.*)estimation_manager(.*)'

# '/uav1/control_manager/mpc_tracker/mpc_reference_debugging'
# '/uav1/control_manager/mpc_tracker/predicted_trajectory_debugging'
# '/uav1/control_manager/tracker_cmd'
'/rosout'
'/rosout_agg'

'(.*)os_cloud_nodelet(.*)'
)

# file's header
filename=`mktemp`
echo "<launch>" > "$filename"
echo "<arg name=\"UAV_NAME\" default=\"\$(env UAV_NAME)\" />" >> "$filename"
echo "<group ns=\"\$(arg UAV_NAME)\">" >> "$filename"

echo -n "<node pkg=\"rosbag\" type=\"record\" name=\"rosbag_record\" output=\"screen\" args=\"-o $path -a" >> "$filename"

# if there is anything to exclude
if [ "${#exclude[*]}" -gt 0 ]; then

  echo -n " -x " >> "$filename"

  # list all the string and separate the with |
  for ((i=0; i < ${#exclude[*]}; i++));
  do
    echo -n "${exclude[$i]}" >> "$filename"
    if [ "$i" -lt "$( expr ${#exclude[*]} - 1)" ]; then
      echo -n "|" >> "$filename"
    fi
  done

fi

echo "\">" >> "$filename"

echo "<remap from=\"~status_msg_out\" to=\"mrs_uav_status/display_string\" />" >> "$filename"
echo "<remap from=\"~data_rate_out\" to=\"~data_rate_MB_per_s\" />" >> "$filename"

# file's footer
echo "</node>" >> "$filename"
echo "</group>" >> "$filename"
echo "</launch>" >> "$filename"

cat $filename
roslaunch $filename
