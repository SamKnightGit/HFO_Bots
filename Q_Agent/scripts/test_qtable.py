#!/usr/bin/env python3.5

import subprocess
import click
import os
import typing
import time
from datetime import datetime
from tqdm import tqdm

qtable_dir = os.path.dirname(os.path.realpath(__file__)) + '/../qtables/2v1_20n_50its_eps01_1000test/' # type: str
qtable_dir = os.path.join(qtable_dir, 'iter_12')

TEST_Q_AGENT_PATH = './example/custom_agents/HFO_Bots/Q_Agent/q_agent_testing.py'

def run_q_table_test(num_agents, num_opponents, trials_per_iteration):
    hfo_process = start_hfo_server(
        num_agents, num_opponents, trials_per_iteration,
    )

    for agent_index in range(0, num_agents):
        start_player(
            num_agents - 1, num_opponents, agent_index,
            trials_per_iteration, in_q_table_path=qtable_dir
        )
        time.sleep(10)

    hfo_process.wait()



def start_hfo_server(num_agents, num_opponents, num_trials, output_file: typing.IO=subprocess.DEVNULL):
    hfo_process = subprocess.Popen(args=['./bin/HFO',
                                         '--offense-agents=' + str(num_agents),
                                         '--defense-npcs=' + str(num_opponents),
                                         '--offense-on-ball', '1',
                                         '--trials', str(num_trials),
                                         '--no-sync', '--no-logging'],
                                   stdout=None)
    time.sleep(15)
    return hfo_process


def start_player(num_teammates, num_opponents, agent_index,
                 trials_per_iteration, log_file:typing.IO=subprocess.DEVNULL,
                 epsilon_value=0.0, learning_rate=0.0, in_q_table_path=None,
                 out_q_table_path=None):

    subprocess.Popen(args=['python', TEST_Q_AGENT_PATH,
                           '--numTeammates=' + str(num_teammates),
                           '--numOpponents=' + str(num_opponents),
                           '--playerIndex=' + str(agent_index + 1),
                           '--qTableDir=' + in_q_table_path + "/q_learner",
                           '--numEpisodes=' + str(trials_per_iteration)],
                     stdout=None)

    time.sleep(10)

if __name__ == '__main__':
    run_q_table_test(2, 1, 5)