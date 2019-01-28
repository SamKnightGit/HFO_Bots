#!/usr/bin/env python3.5

import subprocess
from tqdm import tqdm
import time
import os
import click

TEST_Q_AGENT_PATH = './example/custom_agents/HFO_Bots/Q_Agent/q_agent_testing.py'

qtable_dir = os.path.dirname(os.path.realpath(__file__)) + '/../qtables/2v2_20n_2000its_eps01Q'  # type: str

@click.command()
@click.option('--num_iterations', '-n', default=4,
              help="The number of training and testing iterations to run the bots on.")
@click.option('--trials_per_iteration', '-t', default=50000,
              help="The number of trials (games) of HFO in each iteration")
@click.option('--q_table_directory', '-d', default=qtable_dir,
              help="Path to directory where q tables will be stored.")
def test_2v2(num_iterations, trials_per_iteration, q_table_directory):
    for iteration in tqdm(range(0, num_iterations)):
        out_q_table_path = q_table_directory + "/iter_" + str(iteration)
        hfo_process = subprocess.Popen(args=['./bin/HFO', '--offense-agents=2', '--defense-npcs=2',
                            '--offense-on-ball', '1', '--trials', str(trials_per_iteration),
                            '--no-sync'])

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

        hfo_process.wait()


if __name__ == "__main__":
    test_2v2()