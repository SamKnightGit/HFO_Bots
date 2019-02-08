import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import numpy as np
import os


def plot_test_data():
    save_img_path = os.path.dirname(os.path.abspath(__file__)) + '/../graphs/2v1_20n_5000t_eps01'
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

def plot_data():
    save_img_path = os.path.dirname(os.path.abspath(__file__)) + '/../graphs/2v1_20n_5000t_eps01.png'
    df = get_data_from_files()
    df = df.reset_index()
    df = df.melt('index', var_name='cols', value_name='vals')
    lineplot = sns.lineplot(x='index', y='vals', markers=True, ci="sd", data=df.reset_index())
    fig = lineplot.get_figure()
    fig.savefig(save_img_path)


def get_data_from_files():
    path_to_output_dir = os.path.dirname(os.path.abspath(__file__)) + '/../output/2v1_20n_5000its_eps01/'
    # path_to_output_dir = abs_path_test_dir
    test_dirs = []

    for tdir in os.listdir(path_to_output_dir):
        if 'test_iter' in tdir:
            test_dirs.append(tdir)
    test_dirs.sort(key=file_sort)
    
    goal_percent_lst = []
    frames_lst = []

    for dir_index in range(len(test_dirs)):
        directory = test_dirs[dir_index]
        test_files = []
        dir_path = os.path.join(path_to_output_dir, directory)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            test_files.append(file_path)
        test_files.sort(key=file_sort)
        frames_per_goal, goal_percentage = goal_data_from_files(test_files)
        goal_percent_lst.append(goal_percentage)
        frames_lst.append(frames_per_goal)
    
    trials = get_trials(test_files[0])
    goals_dataframe = pd.DataFrame(columns=[x for x in range(trials, trials*(len(test_files)+1), trials)])
    populate_goals(goals_dataframe, goal_percent_lst)
    goals_dataframe = goals_dataframe.transpose()
    goals_dataframe.columns=['test_run_' + str(x) for x in range(0, len(test_files))] 
    return goals_dataframe


def goal_data_from_files(test_files):
    
    frames_per_goal = []
    goal_percentages = []
    for file_index in range(len(test_files)):
        file = test_files[file_index]
        with open(file, 'r') as fp:
            file_lines = fp.readlines()
        frames_per_goal.append(get_last_value_float(file_lines[13]))
        goal_percent = get_goal_percentage(file_lines[14], file_lines[15])
        goal_percentages.append(goal_percent)
    
    return np.array(frames_per_goal), np.array(goal_percentages)
    
def populate_goals(dataframe, goal_percent_lst):
    for arr_index in range(len(goal_percent_lst)):
        dataframe[dataframe.columns[arr_index]] = goal_percent_lst[arr_index]

def get_dataframe_columns(num_test_runs):
    columns = ['training_runs']
    for i in range(num_test_runs):
        columns.append('test_run_' + str(i))
    return columns
        
def get_last_value_float(line):
    words = line.strip().split(' ')
    return float(words[-1])

def get_goal_percentage(trial_line, goal_line):
    trials = get_last_value_float(trial_line)
    goals = get_last_value_float(goal_line)
    return goals / trials

def get_trials(trial_file):
    with open(trial_file, 'r') as fp:
        file_lines = fp.readlines()
    return int(get_last_value_float(file_lines[14]))

if __name__ == '__main__':
    plot_data()










