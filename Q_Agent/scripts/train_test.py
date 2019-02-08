#!/usr/bin/env python3.5

import subprocess
import click
import os
import typing
import time
from datetime import datetime
from tqdm import tqdm

creation_time = datetime.isoformat(datetime.today())

qtable_dir = os.path.dirname(os.path.realpath(__file__)) + '/../qtables/'  # type: str
qtable_dir += creation_time

output_dir = os.path.dirname(os.path.realpath(__file__)) + '/../output/'   # type: str
output_dir += creation_time

logging_dir = os.path.dirname(os.path.realpath(__file__)) + '/../logs/'
logging_dir += creation_time

TRAIN_Q_AGENT_PATH = './example/custom_agents/HFO_Bots/Q_Agent/q_agent_training.py'
TEST_Q_AGENT_PATH = './example/custom_agents/HFO_Bots/Q_Agent/q_agent_testing.py'


@click.command()
@click.option('--num_agents', '-na', default=2)
@click.option('--num_opponents', '-no', default=2)
@click.option('--num_iterations', '-n', default=4,
              help="The number of training and testing iterations to run the bots on.")
@click.option('--trials_per_iteration', '-t', default=50000,
              help="The number of trials (games) of HFO in each iteration.")
@click.option('--epsilon_start', '-es', default=0.1,
              help="Initial epsilon value.")
@click.option('--epsilon_final', '-ef', default=0.0,
              help="Final epsilon value.")
@click.option('--learning_rate', '-lr', default=0.10,
              help="Learning rate of agents.")
@click.option('--q_table_directory', '-d', default=qtable_dir,
              help="Path to directory where q tables will be stored.")
@click.option('--output_directory', '-o', default=output_dir,
              help="Path to directory where output files will be stored.")
@click.option('--train_only', default=0,
              help="Flag to indicate whether only training will occur.")
@click.option('--num_test_runs', default=0,
              help="Number of test iterations to run for each train iteration."
                   "Automatically sets train_only flag to false when set.")
def train(num_agents, num_opponents, num_iterations, trials_per_iteration,
          epsilon_start, epsilon_final, learning_rate, q_table_directory,
          output_directory, train_only, num_test_runs):

    os.makedirs(q_table_directory, exist_ok=True)

    os.makedirs(output_directory, exist_ok=True)

    os.makedirs(logging_dir, exist_ok=True)

    epsilon_reduce_value = (epsilon_start - epsilon_final) / num_iterations

    if num_test_runs > 0:
        train_only = False

    for iteration in tqdm(range(0, num_iterations)):
        in_q_table_path = None
        if iteration != 0:
            in_q_table_path = q_table_directory + "/iter_" + str(iteration-1) # type: str
        out_q_table_path = q_table_directory + "/iter_" + str(iteration)  # type: str
        os.mkdir(out_q_table_path)

        # initialize q table files
        open(out_q_table_path + '/q_learner1.npy', 'w+').close()
        open(out_q_table_path + '/q_learner2.npy', 'w+').close()

        output_file_name = output_directory + '/train_iter_' + str(iteration) + '.txt'

        # set epsilon value
        epsilon_value = epsilon_start - (iteration * epsilon_reduce_value)

        with open(output_file_name, 'w+') as output_file:
            hfo_process = start_hfo_server(
                num_agents, num_opponents, trials_per_iteration, output_file
            )

            log_file_names = []
            open_log_files = []
            for agent_index in range(0, num_agents):
                logging_file = logging_dir + '/train_iter_' + str(iteration) + \
                               '_player' + str(agent_index + 1) + '.txt'
                log_file_names.append(logging_file)
                open_log_files.append(open(logging_file, 'w+'))

            for agent_index in range(0, num_agents):
                log_file = open_log_files[agent_index]
                start_player(
                    num_agents-1, num_opponents, agent_index, trials_per_iteration, log_file,
                    epsilon_value, learning_rate, in_q_table_path, out_q_table_path
                )
                time.sleep(10)

            hfo_process.wait()
            close_logs(open_log_files)

        clean_keep_lines(logging_dir, log_file_names, 15)
        clean_keep_lines(output_dir, [output_file_name], 20)

    if not train_only:
        test(
            num_agents, num_opponents, num_iterations, trials_per_iteration,
            num_test_runs, q_table_directory, output_directory
        )


def test(num_agents, num_opponents, num_iterations, trials_per_iteration,
         num_test_runs, q_table_directory, output_directory):
    for iteration in tqdm(range(0, num_iterations)):
        output_iteration_dir = os.path.join(output_directory, 'test_iter_' + str(iteration))
        os.makedirs(output_iteration_dir, exist_ok=True)

        in_q_table_path = q_table_directory + "/iter_" + str(iteration)
        for test_run in tqdm(range(0, num_test_runs)):
            output_file_name = os.path.join(output_iteration_dir,
                                            'test_run_' + str(test_run) + '.txt')

            with open(output_file_name, "w+") as output_file:
                hfo_process = start_hfo_server(
                    num_agents, num_opponents, trials_per_iteration, output_file
                )

                for agent_index in range(0, num_agents):
                    start_player(
                        num_agents-1, num_opponents, agent_index,
                        trials_per_iteration, in_q_table_path=in_q_table_path,
                        testing=True
                    )
                    time.sleep(10)

                hfo_process.wait()

            clean_keep_lines(output_dir, [output_file_name], 20)


def clean_keep_lines(logging_directory, logs, num_lines):

    temppath = os.path.join(logging_directory, 'tempfile.txt')

    for log_path in logs:
        with open(temppath, 'w+') as tempfile:
            subprocess.run(['tail', '-n', str(num_lines), log_path], stdout=tempfile)
        subprocess.run(['mv', temppath, log_path])


def start_hfo_server(num_agents, num_opponents, num_trials, output_file: typing.IO=subprocess.DEVNULL):
    hfo_process = subprocess.Popen(args=['./bin/HFO',
                                         '--offense-agents=' + str(num_agents),
                                         '--defense-npcs=' + str(num_opponents),
                                         '--offense-on-ball', '1',
                                         '--trials', str(num_trials),
                                         '--no-sync', '--no-logging'],
                                   stdout=output_file)
    time.sleep(15)
    return hfo_process


def start_player(num_teammates, num_opponents, agent_index,
                 trials_per_iteration, log_file:typing.IO=subprocess.DEVNULL,
                 epsilon_value=0.0, learning_rate=0.0, in_q_table_path=None,
                 out_q_table_path=None, testing=False):
    if testing:
        subprocess.Popen(args=['python', TEST_Q_AGENT_PATH,
                               '--numTeammates=' + str(num_teammates),
                               '--numOpponents=' + str(num_opponents),
                               '--playerIndex=' + str(agent_index + 1),
                               '--qTableDir=' + in_q_table_path + "/q_learner",
                               '--numEpisodes=' + str(trials_per_iteration)],
                         stdout=log_file)

    else:
        if in_q_table_path:
            subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH,
                                   '--numTeammates=' + str(num_teammates),
                                   '--numOpponents=' + str(num_opponents),
                                   '--playerIndex=' + str(agent_index + 1),
                                   '--inQTableDir=' + in_q_table_path + "/q_learner",
                                   '--outQTableDir=' + out_q_table_path + "/q_learner",
                                   '--numEpisodes=' + str(trials_per_iteration),
                                   '--learningRate=' + str(learning_rate),
                                   '--epsilon=' + str(epsilon_value)],
                             stdout=log_file)
        else:
            subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH,
                                   '--numTeammates=' + str(num_teammates),
                                   '--numOpponents=' + str(num_opponents),
                                   '--playerIndex=' + str(agent_index + 1),
                                   '--outQTableDir=' + out_q_table_path + "/q_learner",
                                   '--numEpisodes=' + str(trials_per_iteration),
                                   '--learningRate=' + str(learning_rate),
                                   '--epsilon=' + str(epsilon_value)],
                             stdout=log_file)
    time.sleep(10)


def clean_dir_keep_lines(logging_directory, num_lines):

    files_in_dir = []

    for file in os.listdir(logging_directory):
        files_in_dir.append(os.path.abspath(file))

    clean_keep_lines(logging_directory, files_in_dir, num_lines)


def close_logs(log_file_list):
    for file in log_file_list:
        file.close()

if __name__ == '__main__':
    train()
