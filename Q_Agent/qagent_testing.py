#!/usr/bin/env python3.5

from hfo import *
from os import path
from tqdm import trange
from qplayer import QPlayer
from agent_thread import Agent_Thread
from util.clean import clean_keep_lines
from util.helpers import start_hfo_server

import time


def run_testing(num_testing_set, episodes_per_set, qtable_directory, output_directory, logging_directory,
                num_states, num_actions, num_agents=2, num_opponents=2, port=6000):
    for test_set in trange(0, num_testing_set):
        in_q_table_path = path.join(qtable_directory, 'iter_' + str(test_set))
        output_file_name = os.path.join(output_directory,
                                        'test_iter_' + str(test_set) + '.txt')

        with open(output_file_name, 'w+') as output_file:
            hfo_process = start_hfo_server(
                num_agents, num_opponents, episodes_per_set, output_file, port
            )

            agents = []
            for agent_index in range(0, num_agents):
                input_q_table = None
                if in_q_table_path:
                    input_q_table = path.join(in_q_table_path, 'q_learner' + str(agent_index + 1) + '.npy')

                q_learner = QPlayer(num_states, num_actions, num_agents - 1, num_opponents, port,
                                    episodes_per_set, q_table_in=input_q_table)
                agents.append(q_learner)

            agent_threads = []
            for agent_index in range(0, num_agents):
                logging_file = path.join(logging_directory, 'test_iter_' + str(test_set) +
                                         '_player' + str(agent_index + 1) + '.txt')
                agent_thread = Agent_Thread(agents[agent_index], logging_file)
                agent_threads.append(agent_thread)
                agent_thread.start()
                time.sleep(5)

            for agent in agent_threads:
                agent.join()
            hfo_process.wait()

        clean_keep_lines(output_directory, [output_file_name], 20)