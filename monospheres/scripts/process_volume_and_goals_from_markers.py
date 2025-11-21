#!/usr/bin/env python

import rospy
import rosbag
import sys
import matplotlib.pyplot as plt
from visualization_msgs.msg import MarkerArray
import csv
import numpy as np
import os

OCTOMAP_FREESPACE_TOPICNAME = "/uav1/octomap_local_vis/free_cells_vis_array_throttled"
SPHERES_FREESPACE_TOPICNAME = "/spheremap_freespace"
GOALS_TOPICNAME = "/exploration_goals"
EVAL_CELLSIZE = 2.5
OCTOMAP_CELLSIZE = 0.5

# exp_area_xbounds = [-50, 12]
# exp_area_ybounds = [-40, 10]
# exp_area_zbounds = [0, 7]

def world_to_grid(coord, cellsize):
    """Convert a world coordinate to a discrete grid index."""
    return int(round(coord / cellsize))

def mark_sphere_explored(grid, center, radius, cellsize):
    """Mark cells in a hash map that intersect a given sphere."""
    x_c, y_c, z_c = center
    r_cells = int(radius / cellsize)  # Radius in grid cells

    # # Iterate over grid cells inside the bounding box
    # for x in range(world_to_grid(x_c - radius, cellsize), world_to_grid(x_c + radius, cellsize) + 1):
    #     for y in range(world_to_grid(y_c - radius, cellsize), world_to_grid(y_c + radius, cellsize) + 1):
    #         for z in range(world_to_grid(z_c - radius, cellsize), world_to_grid(z_c + radius, cellsize) + 1):
    #             # Convert grid indices back to world coordinates
    #             x_w = x * cellsize
    #             y_w = y * cellsize
    #             z_w = z * cellsize
                
    #             # Check if the grid cell center is inside the sphere
    #             if (x_w - x_c) ** 2 + (y_w - y_c) ** 2 + (z_w - z_c) ** 2 <= radius ** 2:
    #                 grid[(x, y, z)] = True  # Store in hash map (dictionary)

    # 2D - Iterate over grid cells inside the bounding box
    for x in range(world_to_grid(x_c - radius, cellsize), world_to_grid(x_c + radius, cellsize) + 1):
        for y in range(world_to_grid(y_c - radius, cellsize), world_to_grid(y_c + radius, cellsize) + 1):
            # Convert grid indices back to world coordinates
            x_w = x * cellsize
            y_w = y * cellsize
            
            # Check if the grid cell center is inside the sphere
            if (x_w - x_c) ** 2 + (y_w - y_c) ** 2 <= radius ** 2:
                grid[(x, y)] = True  # Store in hash map (dictionary)


# def compute_free_space(markerarray, cell_size):
    # Get bounds

def process_explored_volume_octomap_gridmarking(bag, cellsize, gridmarking_cellsize, proc_period):
    timestamps = []
    explored_space = []
    start_time = None
    last_proc_time = None
    grid = {}

    message_index = 0
    for topic, msg, t in bag.read_messages(topics=[OCTOMAP_FREESPACE_TOPICNAME ]):
        if last_proc_time is None:
            start_time = t.to_sec()
            last_proc_time = start_time

        if t.to_sec() - last_proc_time > proc_period:
            last_proc_time = t.to_sec()
            explored_volume = 0
            marker_sizes = []

            # EACH MARKER = CUBE ARRAY
            index = 0
            for marker in msg.markers:
                sidelen_mod = (len(msg.markers) - index)
                sidelen = OCTOMAP_CELLSIZE * sidelen_mod
                cube_volume = np.power(sidelen, 3) 
                n_cubes = len(marker.points)

                for pt in marker.points:
                    center = (pt.x, pt.y, pt.z)
                    mark_sphere_explored(grid, center, sidelen, gridmarking_cellsize)
                index += 1

            # Compute explored volume
            explored_cells = len(grid)  # Unique occupied cells
            explored_volume = explored_cells * (gridmarking_cellsize ** 3)
            print("Explored volume: " + str(explored_volume))

            # add data
            timestamps.append(t.to_sec() - start_time)
            explored_space.append(explored_volume)

        message_index += 1

    return timestamps, explored_space


def process_explored_volume_octomap(bag, cellsize, proc_period):
    timestamps = []
    explored_space = []
    start_time = None
    last_proc_time = None

    message_index = 0
    for topic, msg, t in bag.read_messages(topics=[OCTOMAP_FREESPACE_TOPICNAME ]):
        if last_proc_time is None:
            start_time = t.to_sec()
            last_proc_time = start_time

        if t.to_sec() - last_proc_time > proc_period:
            last_proc_time = t.to_sec()
            # print("PROC, index: " + str(message_index))
            explored_volume = 0
            marker_sizes = []
            # EACH MARKER - CUBE ARRAY
            index = 0
            for marker in msg.markers:
                # radius = marker.scale.x / 2.0  # Assuming scale.x is diameter
                sidelen_mod = (len(msg.markers) - index)
                cube_volume = np.power(OCTOMAP_CELLSIZE * sidelen_mod, 3) 
                n_cubes = len(marker.points)

                explored_volume += n_cubes * cube_volume
                index += 1

            # Compute explored volume
            print("Explored volume: " + str(explored_volume))

            # add data
            timestamps.append(t.to_sec() - start_time)
            explored_space.append(explored_volume)

        message_index += 1

    return timestamps, explored_space

def process_explored_volume_spheres(bag, cellsize, proc_period):
    timestamps = []
    explored_space = []
    start_time = None
    last_proc_time = None
    grid = {}
    
    message_index = 0
    for topic, msg, t in bag.read_messages(topics=[SPHERES_FREESPACE_TOPICNAME ]):
        if last_proc_time is None:
            start_time = t.to_sec()
            last_proc_time = start_time

        if t.to_sec() - last_proc_time > proc_period:
            last_proc_time = t.to_sec()
            
            # print("MSG, n markers: " + str(len(msg.markers)))
            # total_radius = sum(marker.scale.x / 2.0 for marker in msg.markers)  # Assuming scale.x is diameter

            # Mark cells in grid as explored
            for marker in msg.markers:
                radius = marker.scale.x / 2.0  # Assuming scale.x is diameter
                center = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
                mark_sphere_explored(grid, center, radius, cellsize)


            # Compute explored volume
            explored_cells = len(grid)  # Unique occupied cells
            explored_volume = explored_cells * (cellsize ** 3)
            print("Explored volume: " + str(explored_volume))
            # print(grid.keys())

            # add data
            timestamps.append(t.to_sec() - start_time)
            explored_space.append(explored_volume)

        message_index += 1

    return timestamps, explored_space

def process_visited_goals(bag, proc_period):
    timestamps = []
    visited_vps = []
    nonvisited_vps = []
    start_time = None
    
    #TODO - fix with proc_period
    process_every_nth = int(1 / proc_period)
    message_index = 0
    for topic, msg, t in bag.read_messages(topics=[GOALS_TOPICNAME]):

        if message_index % process_every_nth == 0:
            print("PROC, index: " + str(message_index))
            if start_time is None:
                start_time = t.to_sec()
            
            print("MSG, n markers: " + str(len(msg.markers)))

            # Mark cells in grid as explored
            n_explored = 0
            n_unexplored = 0
            thresh = 0.8
            for marker in msg.markers:
                if marker.color.r > thresh:
                    n_unexplored += 1
                else:
                    n_explored += 1
                # radius = marker.scale.x / 2.0  # Assuming scale.x is diameter
                # center = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
                # mark_sphere_explored(grid, center, radius, cellsize)
            print("Explored vs Unexplored: " + str(n_explored) + "/" + str(n_unexplored))

            # add data
            timestamps.append(t.to_sec() - start_time)
            visited_vps.append(n_explored)
            nonvisited_vps.append(n_unexplored)

        message_index += 1

    return timestamps, visited_vps, nonvisited_vps

def plot_explored_space(timestamps, explored_space):
    plt.figure()
    plt.plot(timestamps, explored_space, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Explored Free Space (sum of sphere radii)')
    plt.title('Explored Free Space Over Time')
    plt.grid()
    plt.show()

# def save_data_to_csv(filename, timestamps, explored_space, visited_vps, nonvisited_vps):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Time (s)", "Explored Volume", "Visited VPs", "Unvisited VPs"])  # Column headers
#         for t, v, visited, unvisited in zip(timestamps, explored_space, visited_vps, nonvisited_vps):
#             writer.writerow([t, v, visited, unvisited])
#     print(f"Data saved to {filename}")
def save_data_to_csv(filename, timestamps, explored_space):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Explored Volume"])  # Column headers
        for t, v in zip(timestamps, explored_space):
            writer.writerow([t, v])
    print(f"Data saved to {filename}")

def process_and_save_bagfile_data(bagfile_path, proc_period, do_octomap = False, show_res = False):
    try:
        bag = rosbag.Bag(bagfile_path, 'r')
        print(f"Opened bag in file {bagfile_path}")
    except Exception as e:
        print(f"!!!! - Error opening bag file: {e}")
        return False

    endvals = {}
    timestamps = []
    explored = []

    # VOLUME PROC - get octomap OR sphere data
    if not do_octomap:
        print("processing spheremap volume")
        timestamps, explored_space = process_explored_volume_spheres(bag, EVAL_CELLSIZE, proc_period)
    else:
        print("processing octomap volume")
        # timestamps, explored_space = process_explored_volume_octomap(bag, OCTOMAP_CELLSIZE, proc_period)
        timestamps, explored_space = process_explored_volume_octomap_gridmarking(bag, OCTOMAP_CELLSIZE, EVAL_CELLSIZE, proc_period)
    if len(explored_space ) > 0:
        endvals['Total Explored Volume'] = explored_space[-1]

    # SHOW GRAPH
    if show_res:
        plot_explored_space(timestamps, explored_space)


    # GOALS PROC
    # timestamps_vps, visited_vps, nonvisited_vps = process_visited_goals(bag, 1)
    # # plot_explored_space_spheres(timestamps_vps, visited_vps)
    # endvals['Total visited VPs'] = visited_vps[-1]
    # endvals['Final unvisited VPs'] = nonvisited_vps[-1]

    # PRINT FINAL VALS
    print("ENDVALS:")
    for key in endvals.keys():
        print(key)
        print(endvals[key])


    # SAVE DATA
    output_data_path = bagfile_path + "_processed_octomap.txt" if do_octomap else bagfile_path + "_processed_spheres.txt"
    save_data_to_csv(output_data_path, timestamps, explored_space)
    bag.close()

    return True

def process_all_bags_in_path(rootpath, proc_period):
    # GET ALL BAGFILES
    bag_files = []
    for dirpath, dirnames, filenames in os.walk(rootpath):
        for file in filenames:
            if file.endswith(".bag") and not file.endswith(".bag.") and file.count(".bag") == 1:
                full_path = os.path.join(dirpath, file)
                bag_files.append(full_path)
    nfiles = len(bag_files)
    print("N found files: " + str(nfiles))

    # GO THRU ALL
    index = 0
    for bag_file in bag_files:
        index += 1
        print(f"\nProcessing file {index}/{nfiles}: {bag_file}")
        basename = os.path.basename(bag_file)

        do_octomap = False
        if "astar" in basename:
            do_octomap = True

        process_and_save_bagfile_data(bag_file, proc_period, do_octomap)

    return True

def main():
    if len(sys.argv) == 2:
        print("Processing all bags in the run path")
        proc_period = float(sys.argv[1])
        current_path = os.getcwd()
        process_all_bags_in_path(current_path, proc_period)
    elif len(sys.argv) > 2:
        print("Processing one bag:")
        process_and_save_bagfile_data(sys.argv[1], float(sys.argv[2]), do_octomap = False)
    else:
        print("Usage: rosrun your_package script.py <bagfile_path> <proc_period> /OR/ script.py <proc_period> to process all nearby pts")
        return
    
if __name__ == "__main__":
    main()
