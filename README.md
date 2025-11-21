# MonoSpheres
This is the code repository for the MonoSpheres mapping and exploration approach for monocular UAVs.
The method is currently under review, preprint is available at TODO.
## Experiment Videos
Videos from 2 real-world runs using MonoSpheres and runs in the simulated indoor and outdoor environments using both Monospheres and the baseline grid-based mapping and exploration method are available in experiment_videos.mp4. 

## Running the MonoSpheres Examples
This directory containins code for the Monospheres mapping and exploration pipeline, along with a custom version of the OpenVINS package modified for compatibility with the MRS system.
To run the MonoSpheres method example, run `monospheres/tmux_scripts/fireworld_monospheres/tmux.sh`. To run the OctoMap mapper and random explorer approach used as baseline in the paper, run `monospheres/tmux_scripts/fireworld_monospheres/tmux.sh`. 
The parameters used for all the simulation experiments are in the config/mrs_sim.yaml file with comments explaining their purpose.  

To run these scripts, it is neccessary to first install the MRS sytem and build the packages in this repository. Please follow these instructions for installation:

1) Install ROS and the MRS UAV system as per the instructions at https://ctu-mrs.github.io/docs/installation/ either with a native installation or using Apptainer.
2) Create a ROS workspace (if it was not created by the MRS system installation already), add the packages contained in this repository (both the`monospheres/` and `dependecies/` folders) into your workspace's `src/` folder, build the workspace using `catkin_build` and source the workspace by running `source your_workspace_path/devel/setup.bash`
3) Update pip by running `python3 -m pip install --upgrade pip`. Without this, some python packages in the following step might not be found (especially open3d==0.18.0)
4) Create a python virtual environment with the required python packages for the monospheres package using the `./create_python_env.sh` script in the monospheres package.
5) Run `monospheres/tmux_scripts/fireworld_monospheres/tmux.sh`. This will launch a tmux session using the tmux config of the MRS system. If you want to use your own config or the default tmux keybindings, you can delete the line `tmux_options: -f /etc/ctu-mrs/tmux.conf` in `session.yaml`. Otherwise, you will be able to navigate the session using the MRS system tmux keybinds available at https://github.com/ctu-mrs/mrs_cheatsheet?tab=readme-ov-file ). You can close the session using `Ctrl+a, k, enter`

After launching the tmux session, you should see a Gazebo simulation window pop up and a drone should be spawned. Then, after the automatic takeoff, an RVIZ window should open, showing the OpenVINS debug image and the 3D map constructed by MonoSpheres. The `start_maneuver` tab will send a command for the UAV to go 5m forward after takeoff and start the automatic exploration. You can also control the UAV manually by moving to the `status` tmux tab, pressing `Ctrl+R` to switch to the remote mode and the use w,a,s,d,q,e,r,f.  (see https://ctu-mrs.github.io/docs/features/status_tui/ for details). 


## Troubleshooting
- Drone is not spawning - This can be a problem with the jinja2 python package. The code is tested to work with the 2.1.3 version, which can be installed by running `pip install jinja2==2.1.3`. If this does not help, please also update the markupsafe package to the following version: `pip install markupsafe==2.1.3`.
- Drone is spawned but is not taking off - This can happen if a tmux session was not closed properly and a px4 daemon for the hardware API is not destroyed. This can be fixed by closing the session and running `killall px4`.
- Drone has taken off, but nothing is happening:
  - First check if OpenVINS has been initialized. You can check this first by looking at the Rviz `trackhist` image topic where the red text "init" should change into "cam:0" and green SLAM points should appear in the visualization. Also, the messages in the `VINS` tab should change from orange initialization messages into white ones. This might happen if your computer does not have a strong graphics card and if the framerate is low relative to gazebo's real-time-factor (around 30FPS should be available for OpenVINS). This can be solved by slowing down the simulation. In the `session.yaml`, under the `sim_slowdown` tab, uncomment the second line and comment the first one (or call the slowdown manually by `rosrun dynamic_reconfigure dynparam set gazebo max_update_rate 50` once gazebo starts). This should reduce the real_time_factor of gazebo.
  - If OpenVINS is initialized, then check the `navigation` and `exploration_start` tabs. In `navigation`, the monospheres core node should be running and printing runtime statistics. If the map is being built, but the UAV is not exploring, then try running the service in the `exploration_start` tab again to switch the UAV to automatic exploration mode. If you see the map is being built but the UAV is still not exploring, try moving the UAV to another position manually (on different machines, due to using system-time sleeps' the mapping might start after the forward motion is complete, but it is intended to start before the motion, so that the UAV can start exploration in free space). 
