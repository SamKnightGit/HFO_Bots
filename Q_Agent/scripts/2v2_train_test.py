#!/usr/bin/env python3.5

import subprocess
import click
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

creation_time = datetime.isoformat(datetime.today())

qtable_dir = os.path.dirname(os.path.realpath(__file__)) + '/../qtables/'
qtable_dir += creation_time

output_dir = os.path.dirname(os.path.realpath(__file__)) + '/../output/'
output_dir += creation_time

TRAIN_Q_AGENT_PATH = './example/custom_agents/HFO_Bots/Q_Agent/high_level_q_agent.py'
TEST_Q_AGENT_PATH = './example/custom_agents/HFO_Bots/Q_Agent/q_agent_testing.py'
@click.command()
@click.option('--num_iterations', '-n', default=4,
              help="The number of training and testing iterations to run the bots on.")
@click.option('--trials_per_iteration', '-t', default=50000,
              help="The number of trials (games) of HFO in each iteration")
@click.option('--q_table_directory', '-d', default=qtable_dir,
              help="Path to directory where q tables will be stored.")
@click.option('--output_directory', '-o', default=output_dir,
              help="Path to directory where output files will be stored.")
@click.option('--train_only', default=0,
              help="Flag to indicate whether only training will occur.")
def train_2v2(num_iterations, trials_per_iteration, q_table_directory, output_directory, train_only):
    if os.path.exists(q_table_directory):
        sys.exit('Q Table directory already exists, '
                 'please specify another directory or delete the existing one.')
    os.makedirs(q_table_directory)

    if os.path.exists(output_directory):
        sys.exit('Output directory already exists, '
                 'please specify another directory or delete the existing one.')
    os.makedirs(output_directory)

    for iteration in tqdm(range(0, num_iterations)):
        in_q_table_path = None
        if iteration != 0:
            in_q_table_path = q_table_directory + "/iter_" + str(iteration-1)
        out_q_table_path = q_table_directory + "/iter_" + str(iteration)
        os.mkdir(out_q_table_path)

        # initialize q table files
        open(out_q_table_path + '/q_learner1.npy', 'w+').close()
        open(out_q_table_path + '/q_learner2.npy', 'w+').close()

        with open(output_directory + '/train_iter_' + str(iteration) + '.txt', "w+") as output_file:
            HFO_process = subprocess.Popen(args=['./bin/HFO', '--offense-agents=2', '--defense-npcs=2',
                                                 '--offense-on-ball', '1', '--trials', str(trials_per_iteration),
                                                 '--headless'],
                                           stdout=output_file)

        time.sleep(15)

        with open(os.devnull, 'w') as garbage_file:
            if in_q_table_path:
                subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH, '--playerIndex=1',
                                       '--inQTableDir=' + in_q_table_path + "/q_learner",
                                       '--outQTableDir=' + out_q_table_path + "/q_learner",
                                       '--numEpisodes=' + str(trials_per_iteration)],
                                 stdout=garbage_file)
            else:
                subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH, '--playerIndex=1',
                                       '--outQTableDir=' + out_q_table_path + "/q_learner",
                                       '--numEpisodes=' + str(trials_per_iteration)],
                                 stdout=garbage_file)


        time.sleep(10)

        with open(os.devnull, 'w') as garbage_file:
            if in_q_table_path:
                subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH, '--playerIndex=2',
                                       '--inQTableDir=' + in_q_table_path + "/q_learner",
                                       '--outQTableDir=' + out_q_table_path + "/q_learner",
                                       '--numEpisodes=' + str(trials_per_iteration)],
                                 stdout=garbage_file)
            else:
                subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH, '--playerIndex=2',
                                       '--outQTableDir='+out_q_table_path+"/q_learner",
                                       '--numEpisodes='+str(trials_per_iteration)],
                                 stdout=garbage_file)

        HFO_process.wait()

    if not train_only:
        test_2v2(num_iterations, trials_per_iteration, q_table_directory, output_directory)


def test_2v2(num_iterations, trials_per_iteration, q_table_directory, output_directory):
    for iteration in tqdm(range(0, num_iterations)):
        out_q_table_path = q_table_directory + "/iter_" + str(iteration)
        with open(output_directory + '/test_iter_' + str(iteration) + '.txt', "w+") as output_file:
            subprocess.Popen(args=['./bin/HFO', '--offense-agents=2', '--defense-npcs=2',
                                   '--offense-on-ball', '1', '--trials', str(trials_per_iteration),
                                   '--headless'],
                             stdout=output_file)

        time.sleep(15)

        with open(os.devnull, 'w') as garbage_file:
            subprocess.Popen(args=['python', TEST_Q_AGENT_PATH, '--playerIndex=1',
                                   '--qTableDir='+out_q_table_path+"/q_learner",
                                   '--numEpisodes='+str(trials_per_iteration)],
                             stdout=garbage_file)

        time.sleep(10)

        with open(os.devnull, 'w') as garbage_file:
            subprocess.Popen(args=['python', TEST_Q_AGENT_PATH, '--playerIndex=2',
                                   '--qTableDir='+out_q_table_path+"/q_learner",
                                   '--numEpisodes='+str(trials_per_iteration)],
                             stdout=garbage_file)

        time.sleep(10)





if __name__ == '__main__':
    train_2v2()

