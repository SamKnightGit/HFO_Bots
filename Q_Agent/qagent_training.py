#!/usr/bin/env python3.5

from hfo import *
from datetime import datetime
from os import path
from tqdm import trange
from qagent_testing import run_testing
from qlearner import QLearner
from agent_thread import Agent_Thread
from util.clean import clean_keep_lines
from util.helpers import start_hfo_server

import time
import click


creation_time = datetime.isoformat(datetime.today())

base_qtable_dir = path.join(path.dirname(path.realpath(__file__)), 'qtables')
base_qtable_dir = path.join(base_qtable_dir, creation_time)

base_output_dir = path.join(path.dirname(path.realpath(__file__)), 'output')
base_output_dir = path.join(base_output_dir, creation_time)

base_logging_dir = path.join(path.dirname(path.realpath(__file__)), 'logs')
base_logging_dir = path.join(base_logging_dir, creation_time)


@click.command()
@click.option('--num_agents', '-na', default=2)
@click.option('--num_opponents', '-no', default=2)
@click.option('--num_training_set', '-t', default=4,
              help="The number of training sets to run the bots on."
                   "Training set defines when checkpoints of agents internal state is taken.")
@click.option('--episodes_per_set', '-e', default=50000,
              help="The number of episodes (games) of HFO in each trial.")
@click.option('--port', '-p', default=6000,
              help="Port which main HFO server will run on.")
@click.option('--epsilon_start', '-es', default=0.1,
              help="Initial epsilon value.")
@click.option('--epsilon_final', '-ef', default=0.0,
              help="Final epsilon value.")
@click.option('--learning_rate', '-lr', default=0.1,
              help="Learning rate of agents.")
@click.option('--qtable_directory', '-d', default=base_qtable_dir,
              help="Path to directory where q tables will be stored.")
@click.option('--output_directory', '-o', default=base_output_dir,
              help="Path to directory where output files will be stored.")
@click.option('--logging_directory', '-o', default=base_logging_dir,
              help="Path to directory where training logs will be stored.")
@click.option('--train_only', '-to', default=0,
              help="Flag to indicate whether only training will occur.")
@click.option('--num_repeated_runs', '-rr', default=1,
              help="Number of complete training runs to execute.")
@click.option('--num_test_episodes', '-ti', default=0,
              help="Number of test episodes to run for each trial."
                   "Defaults to number train episodes when not set.")
@click.option('--continue_run', '-cr', default=0,
              help="Run to continue from.")
def run_training(num_agents, num_opponents, num_training_set, episodes_per_set, port,
                 epsilon_start, epsilon_final, learning_rate, qtable_directory,
                 output_directory, logging_directory, train_only, num_repeated_runs,
                 num_test_episodes, continue_run):

    num_states = 32 * (54 ** (num_agents - 1))
    num_actions = 2 + (num_agents - 1)
    
    if not continue_run:
        vars_string = "_agents"+str(num_agents)+"_opponents"+str(num_opponents)+ \
                    "_eps"+str(epsilon_start)+"_lr"+str(learning_rate)+"/"
        qtable_directory += vars_string
        output_directory += vars_string
        logging_directory += vars_string

    for run_index in trange(int(continue_run), int(num_repeated_runs)):
        run_qtable_dir = path.join(qtable_directory, 'run_' + str(run_index))
        os.makedirs(run_qtable_dir, exist_ok=True)

        output_dir = path.join(output_directory, 'run_' + str(run_index))
        os.makedirs(output_dir, exist_ok=True)

        logging_dir = path.join(logging_directory, 'run_' + str(run_index))
        os.makedirs(logging_dir, exist_ok=True)

        epsilon_reduce_value = (epsilon_start - epsilon_final) / num_training_set

        if num_test_episodes == 0:
            num_test_episodes = episodes_per_set

        for train_set in trange(0, num_training_set):
            in_q_table_path = None
            if train_set != 0:
                in_q_table_path = path.join(run_qtable_dir, "iter_" + str(train_set-1)) # type: str
            out_q_table_path = path.join(run_qtable_dir, "iter_" + str(train_set))  # type: str
            os.mkdir(out_q_table_path)

            # initialize q table files
            for i in range(1, num_agents + 1):
                open(path.join(out_q_table_path, 'q_learner' + str(i) + '.npy'), 'w+').close()

            output_file_name = path.join(output_dir, 'train_iter_' + str(train_set) + '.txt')

            # set epsilon value
            epsilon_value = epsilon_start - (train_set * epsilon_reduce_value)

            print("Starting HFO Server!")
            with open(output_file_name, 'w+') as output_file:
                hfo_process = start_hfo_server(
                    num_agents, num_opponents, episodes_per_set, output_file, port
                )

                agents = []
                for agent_index in range(0, num_agents):
                    print("Creating Learner: " + str(agent_index))
                    input_q_table = None
                    if in_q_table_path:
                        input_q_table = path.join(in_q_table_path, 'q_learner' + str(agent_index + 1) + '.npy')
                    output_q_table = path.join(out_q_table_path, 'q_learner' + str(agent_index + 1) + '.npy')

                    q_learner = QLearner(num_states, num_actions, num_agents - 1, num_opponents, port,
                                         episodes_per_set, epsilon=epsilon_value, learning_rate=learning_rate,
                                         q_table_in=input_q_table, q_table_out=output_q_table)
                    agents.append(q_learner)

                agent_threads = []
                for agent_index in range(0, num_agents):
                    print("Starting Learner Thread: " + str(agent_index))

                    logging_file = path.join(logging_dir, 'train_iter_' + str(train_set) +
                                             '_player' + str(agent_index + 1) + '.txt')
                    agent_thread = Agent_Thread(agents[agent_index], logging_file)
                    agent_threads.append(agent_thread)
                    agent_thread.start()
                    time.sleep(5)

                for agent in agent_threads:
                    agent.join()
                hfo_process.wait()

            clean_keep_lines(output_dir, [output_file_name], 20)

        if not train_only:
            run_testing(
                num_training_set, num_test_episodes, run_qtable_dir, output_dir, logging_dir,
                num_states, num_actions, num_agents, num_opponents, port
            )


if __name__ == '__main__':
    run_training()