#!/usr/bin/env python3.5

import subprocess
import os
import queue
from typing import List
from threading import Event
import click
import random
import time
import numpy as np
from datetime import datetime
from tqdm import trange
from util import clean_dir_keep_lines
from deep_qnetwork import Global_QNetwork
from deep_qlearner import Deep_QLearner, HL_Deep_QLearner, Discrete_Deep_QLearner
from agent_thread import Learner_Thread
from deep_qagent_testing import run_testing

creation_time = datetime.isoformat(datetime.today())

base_network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn_models')
base_network_dir = os.path.join(base_network_dir, creation_time)

base_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
base_output_dir = os.path.join(base_output_dir, creation_time)

base_logging_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
base_logging_dir = os.path.join(base_logging_dir, creation_time)


@click.command()
@click.option('--port', '-p', default=6000)
@click.option('--seed', '-s', default=random.randint(1,5000))
@click.option('--vary_seed', '-vs', default=False,
              help="Ensure seed changes across parallel games.")
@click.option('--double_q', '-dq', default=False,
              help="Flag indicating whether double q learning should be used")
@click.option('--reward_function', '-rf', default='sparse',
              type=click.Choice(['sparse', 'simple', 'advanced']),
              help="Reward function to be used: sparse, simple or advanced")
@click.option('--state_space', '-ss', default='d',
              type=click.Choice(['d', 'll', 'hl']),
              help="State space choice: d -- discrete, ll -- low level, hl -- high level.")
@click.option('--learning_rate', '-lr', default=0.0001,
              help="Learning rate of network.")
@click.option('--epsilon_start', '-es', default=0.80,
              help="Initial epsilon value.")
@click.option('--epsilon_final', '-ef', default=0.0,
              help="Final epsilon value.")
@click.option('--num_agents', '-tm', default=2)
@click.option('--num_opponents', '-op', default=2)
@click.option('--iterations_per_trial', '-i', default=5)
@click.option('--num_trials', '-t', default=1)
@click.option('--num_parallel_games', '-pg', default=3)
@click.option('--save_network_dir', '-ns', default=base_network_dir)
@click.option('--load_network_dir', '-nl', default=base_network_dir)
@click.option('--logging_dir', '-l', default=base_logging_dir)
@click.option('--output_dir', '-o', default=base_output_dir)
@click.option('--train_only', '-to', default=0,
              help="Flag to indicate whether only training will occur.")
@click.option('--continue_training', '-ct', default=0,
              help="Flag to indicate whether training should be continued"
                   "from a previous point.")
@click.option('--start_iteration', '-sn', default=0,
              help="Iteration to start training from if continuing.")
@click.option('--num_repeated_runs', '-rr', default=1,
              help="Number of complete training runs to execute.")
@click.option('--num_test_iterations', '-ti', default=0,
              help="Number of test iterations to run for each trial."
                   "Defaults to number train iterations when not set.")
def run_training(port, seed, vary_seed, double_q, reward_function, state_space, learning_rate,
                 epsilon_start, epsilon_final, num_agents, num_opponents, iterations_per_trial,
                 num_trials, num_parallel_games, save_network_dir, load_network_dir,
                 logging_dir, output_dir, train_only, continue_training, start_iteration,
                 num_repeated_runs, num_test_iterations):

    if not continue_training:
        vars_string = "_agents"+str(num_agents)+"_opponents"+str(num_opponents)+ \
                      "_eps"+str(epsilon_start)+"_lr"+str(learning_rate)+ \
                      "_pg"+str(num_parallel_games)+"_doubleq" + str(double_q) + \
                      "_"+str(state_space)+"/"
        load_network_dir += vars_string
        save_network_dir += vars_string
        output_dir += vars_string
        logging_dir += vars_string
    
    for run_index in trange(int(num_repeated_runs)):
        save_net_parent_dir = os.path.join(save_network_dir, 'run_' + str(run_index))
        load_net_dir = os.path.join(load_network_dir, 'run_' + str(run_index))
        log_parent_dir = os.path.join(logging_dir, 'run_' + str(run_index))
        out_parent_dir = os.path.join(output_dir, 'run_' + str(run_index))

        experience_list = []

        epsilon_reduce_value = (epsilon_start - epsilon_final) / num_trials

        if num_test_iterations > 0:
            train_only = False

        num_test_iterations = iterations_per_trial

        num_teammates = num_agents - 1

        if state_space == 'll':
            state_dimensions = 59 + (9 * num_teammates) + (9 * num_opponents)
        elif state_space == 'hl':
            state_dimensions = 12 + (6 * num_teammates) + (3 * num_opponents)
        else: # discrete state space
            state_dimensions = 4 + (4 * num_teammates)

        if continue_training:
            network_path = os.path.join(
                load_net_dir, 'iter_' + str(start_iteration-1), 'main_net.h5')
            global_network = Global_QNetwork(
                load_location=network_path
            )
        else:
            global_network = Global_QNetwork(
                state_dimensions, learning_rate, num_teammates
            )

        for iteration in trange(int(start_iteration), int(num_trials)):

            save_net_dir = os.path.join(save_net_parent_dir, 'iter_' + str(iteration))
            os.makedirs(save_net_dir, exist_ok=True)

            global_network.set_save_location(save_net_dir)

            log_dir = os.path.join(log_parent_dir, 'iter_' + str(iteration))
            os.makedirs(log_dir, exist_ok=True)

            out_dir = os.path.join(out_parent_dir, 'iter_' + str(iteration))
            os.makedirs(out_dir, exist_ok=True)

            epsilon_value = epsilon_start - (iteration * epsilon_reduce_value)

            deep_learners = []
            for game_index in range(0, num_parallel_games):
                print("Connecting for game: " + str(game_index))
                unique_port = port + 5 * game_index

                for agent_index in range(0, int(num_agents)):
                    if state_space == 'hl':
                        deep_learner = HL_Deep_QLearner(
                            global_network, reward_function, experience_list, unique_port, double_q,
                            learning_rate, epsilon_value, iterations_per_trial, num_teammates, num_opponents
                        )
                    elif state_space == 'll':
                        deep_learner = Deep_QLearner(
                            global_network, reward_function, experience_list, unique_port, double_q,
                            learning_rate, epsilon_value, iterations_per_trial, num_teammates, num_opponents
                        )
                    else:
                        deep_learner = Discrete_Deep_QLearner(
                            global_network, reward_function, experience_list, unique_port, double_q,
                            learning_rate, epsilon_value, iterations_per_trial, num_teammates, num_opponents
                        )
                    print("Agent " + str(agent_index) +
                        " connected for game: " + str(game_index))
                    deep_learners.append(deep_learner)

            hfo_processes = []
            
            for game_index in range(0, num_parallel_games):
                unique_port = port + 5 * game_index # consecutive ports does not work well
                unique_seed = seed
                if vary_seed:
                    unique_seed += game_index

                output_file = os.path.join(out_dir, 'game_' + str(game_index) + '.txt')
                with open(output_file, 'w+') as outfile:
                    hfo_game = subprocess.Popen(args=['./bin/HFO',
                                                    '--offense-agents='+str(num_agents),
                                                    '--defense-npcs='+str(num_opponents),
                                                    '--offense-on-ball', '1',
                                                    '--trials', str(iterations_per_trial),
                                                    '--port', str(unique_port),
                                                    '--seed', str(unique_seed),
                                                    '--headless',
                                                    '--no-logging'],
                                                stdout=outfile,
                                                stderr=outfile)

                hfo_processes.append(hfo_game)

            time.sleep(15)

            learners = []
            finished_list = [0] * len(deep_learners)  # type: List[int]
            for learner_index in range(0, len(deep_learners)):
                print("Learner " + str(learner_index) + " started.")
                log_file = os.path.join(log_dir, 'learner_' + str(learner_index) + '.txt')
                learner_thread = Learner_Thread(
                    deep_learners[learner_index], learner_index, finished_list, log_file
                )
                learners.append(learner_thread)
                learner_thread.start()
                time.sleep(5)

            while 0 in finished_list:
                experience_list_copy = experience_list.copy()
                states = np.array([x[0][0] for x in experience_list_copy])
                targets = np.array([x[1][0] for x in experience_list_copy])
                print("\n***********************\n")
                print(targets)
                print("Length of arrays {0}".format(len(states)))
                print("\n***********************\n")
                if states.any():
                    global_network.net.fit(states, targets, batch_size=len(states), verbose=0)
                experience_list.clear()
                # accumulate experiences for one second
                time.sleep(1)

            for learner in learners:
                learner.join()

            # Wait for all servers to close
            for hfo_process in hfo_processes:
                hfo_process.wait()

            clean_dir_keep_lines(out_dir, 20)
            global_network.save_network()

        if not train_only:
            run_testing(
                load_net_dir, out_parent_dir, port, num_agents, num_opponents,
                num_trials, num_test_iterations, state_space
            )


if __name__ == '__main__':
    run_training()


