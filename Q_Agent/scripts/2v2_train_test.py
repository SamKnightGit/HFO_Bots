#!/usr/bin/env python3.5

import subprocess
import click
import os
import sys
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
@click.option('--num_iterations', '-n', default=4,
              help="The number of training and testing iterations to run the bots on.")
@click.option('--trials_per_iteration', '-t', default=50000,
              help="The number of trials (games) of HFO in each iteration.")
@click.option('--epsilon_start', '-es', default=0.1,
              help="Initial epsilon value.")
@click.option('--epsilon_final', '-ef', default=0.0,
              help="Final epsilon value.")
@click.option('--q_table_directory', '-d', default=qtable_dir,
              help="Path to directory where q tables will be stored.")
@click.option('--output_directory', '-o', default=output_dir,
              help="Path to directory where output files will be stored.")
@click.option('--train_only', default=0,
              help="Flag to indicate whether only training will occur.")
def train_2v2(num_iterations, trials_per_iteration, epsilon_start, epsilon_final,
              q_table_directory, output_directory, train_only):
    if os.path.exists(q_table_directory):
        sys.exit('Q Table directory already exists, '
                 'please specify another directory or delete the existing one.')
    os.makedirs(q_table_directory)

    if os.path.exists(output_directory):
        sys.exit('Output directory already exists, '
                 'please specify another directory or delete the existing one.')
    os.makedirs(output_directory)

    os.makedirs(logging_dir, exist_ok=True)


    epsilon_reduce_value = (epsilon_start - epsilon_final) / num_iterations

    for iteration in tqdm(range(0, num_iterations)):
        in_q_table_path = None
        if iteration != 0:
            in_q_table_path = q_table_directory + "/iter_" + str(iteration-1)
        out_q_table_path = q_table_directory + "/iter_" + str(iteration)  # type: str
        os.mkdir(out_q_table_path)

        # initialize q table files
        open(out_q_table_path + '/q_learner1.npy', 'w+').close()
        open(out_q_table_path + '/q_learner2.npy', 'w+').close()

        logging_file_name_1 = logging_dir + '/train_iter_' + str(iteration) + '_player1.txt'
        logging_file_name_2 = logging_dir + '/train_iter_' + str(iteration) + '_player2.txt'

        output_file_name = output_directory + '/train_iter_' + str(iteration) + '.txt'


        # set epsilon value
        epsilon_value = epsilon_start - (iteration * epsilon_reduce_value)

        with open(output_file_name, "w+") as output_file, \
                open(logging_file_name_1, 'w+') as logging_file_1, \
                open(logging_file_name_2, 'w+') as logging_file_2:
            hfo_process = subprocess.Popen(args=['./bin/HFO', '--offense-agents=2', '--defense-npcs=2',
                                                 '--offense-on-ball', '1', '--trials', str(trials_per_iteration),
                                                 '--headless', '--no-logging'],
                                           stdout=output_file)

            time.sleep(15)

            if in_q_table_path:
                subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH, '--playerIndex=1',
                                       '--inQTableDir=' + in_q_table_path + "/q_learner",
                                       '--outQTableDir=' + out_q_table_path + "/q_learner",
                                       '--numEpisodes=' + str(trials_per_iteration),
                                       '--epsilon=' + str(epsilon_value)],
                                 stdout=logging_file_1)
            else:
                subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH, '--playerIndex=1',
                                       '--outQTableDir=' + out_q_table_path + "/q_learner",
                                       '--numEpisodes=' + str(trials_per_iteration),
                                       '--epsilon=' + str(epsilon_value)],
                                 stdout=logging_file_1)

            time.sleep(10)

            if in_q_table_path:
                subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH, '--playerIndex=2',
                                       '--inQTableDir=' + in_q_table_path + "/q_learner",
                                       '--outQTableDir=' + out_q_table_path + "/q_learner",
                                       '--numEpisodes=' + str(trials_per_iteration),
                                       '--epsilon=' + str(epsilon_value)],
                                 stdout=logging_file_2)
            else:
                subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH, '--playerIndex=2',
                                       '--outQTableDir='+out_q_table_path+"/q_learner",
                                       '--numEpisodes='+str(trials_per_iteration),
                                       '--epsilon=' + str(epsilon_value)],
                                 stdout=logging_file_2)

            hfo_process.wait()

        clean_keep_lines(logging_dir, [logging_file_name_1, logging_file_name_2], 15)
        clean_keep_lines(output_dir, [output_file_name], 20)

    if not train_only:
        test_2v2(num_iterations, trials_per_iteration, q_table_directory, output_directory)


def test_2v2(num_iterations, trials_per_iteration, q_table_directory, output_directory):
    for iteration in tqdm(range(0, num_iterations)):
        in_q_table_path = q_table_directory + "/iter_" + str(iteration)

        if not os.path.exists(logging_dir):
            os.mkdir(logging_dir)

        logging_file_name_1 = logging_dir + '/test_iter_' + str(iteration) + '_player1.txt'
        logging_file_name_2 = logging_dir + '/test_iter_' + str(iteration) + '_player2.txt'

        output_file_name = output_directory + '/test_iter_' + str(iteration) + '.txt'

        with open(output_file_name, "w+") as output_file, \
            open(logging_file_name_1, 'w+') as logging_file_1, \
            open(logging_file_name_2, 'w+') as logging_file_2:

            hfo_process = subprocess.Popen(args=['./bin/HFO', '--offense-agents=2', '--defense-npcs=1',
                                   '--offense-on-ball', '1', '--trials', str(trials_per_iteration),
                                   '--headless', '--no-logging'],
                             stdout=output_file)

            time.sleep(15)


            subprocess.Popen(args=['python', TEST_Q_AGENT_PATH, '--playerIndex=1',
                                   '--qTableDir='+in_q_table_path+"/q_learner",
                                   '--numEpisodes='+str(trials_per_iteration)],
                             stdout=logging_file_1)

            time.sleep(10)


            subprocess.Popen(args=['python', TEST_Q_AGENT_PATH, '--playerIndex=2',
                                   '--qTableDir='+in_q_table_path+"/q_learner",
                                   '--numEpisodes='+str(trials_per_iteration)],
                             stdout=logging_file_2)

            hfo_process.wait()

        clean_keep_lines(logging_dir, [logging_file_name_1, logging_file_name_2], 15)
        clean_keep_lines(output_dir, [output_file_name], 20)


def clean_keep_lines(logging_directory, logs, num_lines):

    temppath = logging_directory + '/tempfile.txt'

    for log_path in logs:
        with open(temppath, 'w+') as tempfile:
            subprocess.run(['tail', '-n', str(num_lines), log_path], stdout=tempfile)
        subprocess.run(['mv', temppath, log_path])


def clean_dir_keep_lines(logging_directory, num_lines):

    files_in_dir = []

    for file in os.listdir(logging_directory):
        files_in_dir.append(os.path.abspath(file))

    clean_keep_lines(logging_directory, files_in_dir, num_lines)


if __name__ == '__main__':
    train_2v2()
