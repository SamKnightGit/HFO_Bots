from os.path import join, dirname, abspath
import os
import numpy as np


def goal_data_from_files(test_files):
    frames_per_goal = []
    goal_percentages = []
    for file_index in range(len(test_files)):
        file = test_files[file_index]
        with open(file, 'r') as fp:
            file_lines = fp.readlines()
        try:
            frames_per_goal.append(get_last_value_float(file_lines[13]))
            goal_percent = get_goal_percentage(file_lines[14], file_lines[15])
            goal_percentages.append(goal_percent)
        except (IndexError, ValueError):
            continue

    return frames_per_goal, goal_percentages


def get_last_value_float(line):
    words = line.strip().split(' ')
    return float(words[-1])


def get_goal_percentage(trial_line, goal_line):
    trials = get_last_value_float(trial_line)
    goals = get_last_value_float(goal_line)
    try:
        return goals / trials
    except ZeroDivisionError:
        return 0


if __name__ == '__main__':
    output_dir = join(dirname(dirname(abspath(__file__))), 'output')
    for output in os.listdir(output_dir):
        print("*************************")
        print(output)
        max_goal = 0
        frames_to_goal = 0
        for run in os.listdir(join(output_dir, output)):
            files = []
            for file in os.listdir(join(output_dir, output, run)):
                if 'test_iter' in file:
                    files.append(join(output_dir, output, run, file))
            frames_per_goal, goal_percentages = goal_data_from_files(files)
            if len(goal_percentages) == 0:
                continue
            max_goal_percentage = max(goal_percentages)
            if max_goal_percentage > max_goal:
                max_goal = max_goal_percentage
                goal_percentages = np.array(goal_percentages)
                frames_to_goal = frames_per_goal[np.argmax(goal_percentages)]
        print(max_goal)
        print(frames_to_goal)
        print("*************************")

