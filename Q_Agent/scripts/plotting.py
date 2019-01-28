import matplotlib.pyplot as plt
import numpy as np

import os

def plot_test_data():
    save_img_path = os.path.dirname(os.path.abspath(__file__)) + '/../graphs/2v2_20n_2000t_eps01_fpg'
    frames_per_goal, goal_percentages, trials = get_data_from_files()
    print(goal_percentages)
    print(frames_per_goal)
    num_experiments = len(frames_per_goal)
    x_vals = [trials * x for x in range(1, num_experiments+1)]

    plt.subplot(2,1,1)
    plt.plot(x_vals, goal_percentages, 'b+')
    plt.axis([trials, trials*num_experiments, 0.5, 1.0])
    plt.xticks(x_vals[1::2])
    plt.ticklabel_format(axis='x', style='sci', scilimits=(4,4))
    plt.xlabel("Number of episodes")
    plt.ylabel("% goal scoring episodes")

    plt.subplot(2,1,2)
    plt.plot(x_vals, frames_per_goal, 'b+')
    plt.axis([trials, trials * num_experiments, 60, 110])
    plt.xticks(x_vals[1::2])
    plt.ticklabel_format(axis='x', style='sci', scilimits=(4,4))
    plt.xlabel("Number of episodes")
    plt.ylabel("Avg. frames per goal")

    plt.tight_layout()
    plt.savefig(save_img_path)

def file_sort(file_name):
    number_file_ext = file_name.split('_')[-1]
    return int(number_file_ext.split('.')[0])

def get_data_from_files():
    path_to_output_dir = os.path.dirname(os.path.abspath(__file__)) + '/../output/2v2_20n_2000its_eps01/'
    # path_to_output_dir = abs_path_test_dir
    test_files = []

    for file in os.listdir(path_to_output_dir):
        if 'test_iter' in file:
            test_files.append(file)
    test_files.sort(key=file_sort)

    frames_per_goal = []
    goal_percentages = []
    for file in test_files:
        file_path = path_to_output_dir + file
        with open(file_path, 'r') as fp:
            file_lines = fp.readlines()
        frames_per_goal.append(get_last_value_float(file_lines[13]))
        goal_percent, trials = get_goal_percentage(file_lines[14], file_lines[15])
        goal_percentages.append(goal_percent)

    return np.array(frames_per_goal), np.array(goal_percentages), trials

def get_last_value_float(line):
    words = line.strip().split(' ')
    return float(words[-1])

def get_goal_percentage(trial_line, goal_line):
    trials = get_last_value_float(trial_line)
    goals = get_last_value_float(goal_line)
    return goals / trials, trials

if __name__ == '__main__':
    plot_test_data()










