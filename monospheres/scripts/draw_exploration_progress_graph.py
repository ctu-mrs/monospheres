import csv
import sys
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

MAX_EXPERIMENT_TIME = 20 * 60

def read_data_from_txt(filename):
    # timestamps, explored_space, visited_vps, nonvisited_vps = [], [], [], []
    timestamps, explored_space = [], [] 

    with open(filename, mode='r') as file:
        next(file)  # skip header
        for line in file:
            # print(line)
            # t, v, visited, unvisited = line.strip().split(",")
            vals = line.strip().split(",")
            t = vals[0]
            v = vals[1]
            timestamps.append(float(t))
            explored_space.append(float(v))
            # visited_vps.append(int(visited))
            # nonvisited_vps.append(int(unvisited))

    # return timestamps, explored_space, visited_vps, nonvisited_vps
    return timestamps, explored_space

def plot_exploration_progress_simple(datas):

    for experiment in datas:
        # plt.plot(experiment['timestamps'], experiment['explored_volumes'], marker='o', linestyle='-', label = experiment['name'])

        timestamps = experiment['timestamps']
        explored_volumes = experiment['explored_volumes']
        
        # Plot the normal trajectory
        plt.plot(timestamps, explored_volumes, marker='o', linestyle='-', label=experiment['name'])
        
    plt.xlabel('Time [s]')
    plt.ylabel('Explored Area [columns]')
    # plt.title('explored free space over time')
    plt.legend()
    plt.grid()
    plt.show()

def plot_exploration_progress_clustered(datas):

    my_dpi = 100
    plt.figure(figsize=(650/my_dpi, 650/my_dpi), dpi=my_dpi)

    colors = {}
    colors['MonoSpheres'] = 'green'
    colors['OM-RandomExplo'] = 'blue'
    colors['MS-NoOFS'] = 'red'
    colors['MS-RandomExplo'] = 'orange'
    labeled = set()

    for experiment in datas:
        clr = 'black'
        alpha = 0.7
        if experiment['method'] in colors.keys():
            clr = colors[experiment['method']]
        if experiment['method'] in labeled:
            plt.plot(experiment['timestamps'], experiment['explored_volumes'], linestyle='-', color = clr, alpha=alpha)
        else:
            plt.plot(experiment['timestamps'], experiment['explored_volumes'], linestyle='-', color = clr, label=experiment['method'], alpha=alpha)
            labeled.add(experiment['method'])

        timestamps = experiment['timestamps']
        explored_volumes = experiment['explored_volumes']

        # Check if experiment ended before max time (crash case)
        # if timestamps[-1] < MAX_EXPERIMENT_TIME - 10:
        if True:
            print("ENDED SOONER")
            plt.plot(
                timestamps[-1], 
                explored_volumes[-1], 
                marker='X', 
                markersize=10, 
                markeredgewidth=0.01,
                color=clr,
                # label=f"{experiment['name']} crash"
            )
    plt.xlabel('Time [s]', fontsize = 18)
    plt.ylabel('Explored Area [m^2]', fontsize = 18)
    plt.title('Exploration Progress', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.show()

def extract_table_data(datas):
    # todo - print these values for each method: number of experiments, average volume explored, max volume explored, number of crashes across all experiments
    method_stats = defaultdict(lambda: {
        'num_experiments': 0,
        'total_volume': 0.0,
        'max_volume': 0.0,
        'best_volume_rosbag': 'none',
        'num_crashes': 0
    })

    print("max experiment min: " + str(MAX_EXPERIMENT_TIME / 60))
    for experiment in datas:
        method = experiment['method']
        final_volume = experiment['explored_volumes'][-1] if experiment['explored_volumes'] else 0.0
        maxtime = experiment['total_time']
        
        method_stats[method]['num_experiments'] += 1
        method_stats[method]['total_volume'] += final_volume

        # print("Method " + method + " MAXTIME: " + str(maxtime) + " Total Volume: " + str(final_volume) + " Exper: " + experiment['name']) 
        print("Method " + method + " MAXTIME: " + str(maxtime) + " Total Volume: " + str(final_volume)) 
        print( " Exper: " + experiment['name']) 
        print("---") 

        # method_stats[method]['max_volume'] = max(method_stats[method]['max_volume'], final_volume)
        if method_stats[method]['max_volume'] < final_volume:
            method_stats[method]['max_volume'] = final_volume
            method_stats[method]['best_volume_rosbag'] = experiment['name']
        if experiment['crash']:
            method_stats[method]['num_crashes'] += 1

    print("\n=== Table Summary ===")
    for method, stats in method_stats.items():
        avg_volume = stats['total_volume'] / stats['num_experiments'] if stats['num_experiments'] > 0 else 0.0
        print(f"Method: {method}")
        print(f"  Number of Experiments: {stats['num_experiments']}")
        print(f"  Average Volume Explored: {avg_volume:.2f}")
        print(f"  Max Volume Explored: {stats['max_volume']:.2f}")
        print(f"  Max Volume Rosbag {stats['best_volume_rosbag']:s}")
        print(f"  Number of Crashes: {stats['num_crashes']}")
        print("---------------------------")

def extract_datas(files):
    datas = []

    column_volume = (2.5 ** 3)
    # area_x_min =  -50
    # area_x_max =  12
    # area_y_min =  -40
    # area_y_max =  10
    # area_max = (area_x_max - area_x_min) * (area_y_max - area_y_min)
    # column_area = 2.5 ** 2.5

    for i in range(len(files)):
        filename = files[i]
        print("reading experiment " + str(i) + "from file: " + filename)
        timestamps, explored_volumes = read_data_from_txt(filename)
        experiment = {}
        experiment['timestamps'] = timestamps
        experiment['total_time'] = timestamps[-1]
        experiment['explored_volumes'] = (np.array(explored_volumes) / 2.5).tolist()
        # experiment['explored_volumes'] = (np.array(explored_volumes) / column_volume).tolist()
        # experiment['explored_volumes'] = ((column_area *(np.array(explored_volumes) / column_volume)) / area_max).tolist()

        # CUT EXPERIMENT LEN (e.g. for when rosbag recording was left running for longer)
        if timestamps[-1] > MAX_EXPERIMENT_TIME:
            endval_index = np.argmax(np.array(timestamps) > MAX_EXPERIMENT_TIME)
            experiment['timestamps'] = experiment['timestamps'][:endval_index]
            experiment['explored_volumes'] = experiment['explored_volumes'][:endval_index] # end the data at this time
            experiment['explored_volumes'][-1] = experiment['explored_volumes'][-2] # set this so that the drawn line goes a bit behind the max time for drawing purposes


        experiment['method'] = 'unknown'
        if "noofs" in filename:
            experiment['method'] = 'MS-NoOFS'
        elif "ofs" in filename:
            experiment['method'] = 'MonoSpheres'
        elif "astar" in filename or "gridbased" in filename:
            experiment['method'] = 'OM-RandomExplo'
        elif "monorandom" in filename:
            experiment['method'] = 'MS-RandomExplo'

        experiment['crash'] = "crash" in filename

        experiment['name'] = filename

        datas.append(experiment)
    return datas

def plot_progress_from_all_files(rootpath):

    # get all data files
    data_files = []
    for dirpath, dirnames, filenames in os.walk(rootpath):
        for file in filenames:
            if file.endswith(".txt"):
                full_path = os.path.join(dirpath, file)
                data_files.append(full_path)
    nfiles = len(data_files)
    print("n found files: " + str(nfiles))

    # go thru all and extract data from file
    datas = []
    datas = extract_datas(data_files)
    # index = 0
    # for data_file in data_files:
    #     index += 1
    #     print(f"\nprocessing file {index}/{nfiles}: {data_file}")

    #     do_octomap = false
    #     # todo - draw all into graph

    extract_table_data(datas)

    plot_exploration_progress_clustered(datas)

    

    return True

def main():
    if len(sys.argv) > 1:
        files = []
        print("processing given bagfiles, simple")
        for i in range(len(sys.argv) - 1):
            files += sys.argv[i+1]
        datas = extract_datas(files)

        print(f"read {len(datas)} experiments")
        plot_exploration_progress_simple(datas)
    else:
        rootpath = os.getcwd()
        print("processing all data in this folder: " + rootpath)
        plot_progress_from_all_files(rootpath)

if __name__ == "__main__":
    main()
