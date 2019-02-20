#!/usr/bin/env python3.5

import subprocess
import os
import queue
from typing import List
from threading import Event
import click
import random
import time
from datetime import datetime
from tqdm import trange
from util import clean_dir_keep_lines, clean_keep_lines
from deep_qnetwork import Global_QNetwork
from deep_qlearner import Deep_QLearner, HL_Deep_QLearner
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
@click.option('--high_level_state_space', '-h', default=0,
              help="Flag determining whether high level state space"
                   "is used, instead of low level.")
@click.option('--learning_rate', '-lr', default=0.90,
              help="Learning rate of network.")
@click.option('--epsilon_start', '-es', default=0.10,
              help="Initial epsilon value.")
@click.option('--epsilon_final', '-ef', default=0.0,
              help="Final epsilon value.")
@click.option('--num_agents', '-tm', default=2)
@click.option('--num_opponents', '-op', default=2)
@click.option('--trials_per_iteration', '-t', default=5)
@click.option('--num_iterations', '-n', default=1)
@click.option('--num_parallel_games', '-pg', default=3)
@click.option('--save_network_dir', '-ns', default=base_network_dir)
@click.option('--load_network_dir', '-nl', default=base_network_dir)
@click.option('--logging_dir', '-l', default=base_logging_dir)
@click.option('--output_dir', '-o', default=base_output_dir)
@click.option('--train_only', '-to', default=0,
              help="Flag to indicate whether only training will occur.")
@click.option('--num_repeated_runs', '-rr', default=1,
              help="Number of complete training runs to execute.")
@click.option('--num_test_trials', '-tt', default=0,
              help="Number of test trials to run for each iteration."
                   "Defaults to number train iterations when not set.")
def run_training(port, seed, high_level_state_space, learning_rate, epsilon_start,
                 epsilon_final, num_agents, num_opponents, trials_per_iteration,
                 num_iterations, num_parallel_games, save_network_dir, load_network_dir,
                 logging_dir, output_dir, train_only, num_repeated_runs, num_test_trials):
    
    vars_string = "_agents"+str(num_agents)+"_opponents"+str(num_opponents)+ \
                  "_eps"+str(epsilon_start)+"_lr"+str(learning_rate)+"/"
    load_network_dir += vars_string
    save_network_dir += vars_string
    output_dir += vars_string
    logging_dir += vars_string
    
    for run_index in trange(int(num_repeated_runs)):
        save_net_parent_dir = os.path.join(save_network_dir, 'run_' + str(run_index))
        load_net_dir = os.path.join(load_network_dir, 'run_' + str(run_index))
        log_parent_dir = os.path.join(logging_dir, 'run_' + str(run_index))
        out_parent_dir = os.path.join(output_dir, 'run_' + str(run_index))

        experience_queue = queue.Queue()

        epsilon_reduce_value = (epsilon_start - epsilon_final) / num_iterations

        if num_test_trials > 0:
            train_only = False

        num_test_trials = trials_per_iteration

        num_teammates = num_agents - 1

        if high_level_state_space:
            state_dimensions = 11 + (6 * num_teammates) + (3 * num_opponents)
        else:
            state_dimensions = 59 + (9 * num_teammates) + (9 * num_opponents)

        global_network = Global_QNetwork(
            state_dimensions, learning_rate, num_teammates
        )

        for iteration in trange(int(num_iterations)):

            save_net_dir = os.path.join(save_net_parent_dir, 'iter_' + str(iteration))
            os.makedirs(save_net_dir, exist_ok=True)

            global_network.set_save_location(save_net_dir)

            log_dir = os.path.join(log_parent_dir, 'iter_' + str(iteration))
            os.makedirs(log_dir, exist_ok=True)

            out_dir = os.path.join(out_parent_dir, 'iter_' + str(iteration))
            os.makedirs(out_dir, exist_ok=True)

            epsilon_value = epsilon_start - (iteration * epsilon_reduce_value)

            deep_learners = []
            update_event = Event()
            for game_index in range(0, num_parallel_games):
                print("Connecting for game: " + str(game_index))
                unique_port = port + 5 * game_index

                for agent_index in range(0, int(num_agents)):
                    if high_level_state_space:
                        deep_learner = HL_Deep_QLearner(
                            global_network, update_event, experience_queue, unique_port, learning_rate,
                            epsilon_value, trials_per_iteration, num_teammates, num_opponents
                        )
                    else:
                        deep_learner = Deep_QLearner(
                            global_network, update_event, experience_queue, unique_port, learning_rate,
                            epsilon_value, trials_per_iteration, num_teammates, num_opponents
                        )
                    print("Agent " + str(agent_index) +
                        " connected for game: " + str(game_index))
                    deep_learners.append(deep_learner)

            hfo_processes = []
            
            for game_index in range(0, num_parallel_games):
                unique_port = port + 5 * game_index # consecutive ports does not work well
                unique_seed = seed + game_index

                output_file = os.path.join(out_dir, 'game_' + str(game_index) + '.txt')
                with open(output_file, 'w+') as outfile:
                    hfo_game = subprocess.Popen(args=['./bin/HFO',
                                                    '--offense-agents='+str(num_agents),
                                                    '--defense-npcs='+str(num_opponents),
                                                    '--offense-on-ball', '1',
                                                    '--trials', str(trials_per_iteration),
                                                    '--port', str(unique_port),
                                                    '--seed', str(unique_seed),
                                                    '--headless'],
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
                try:
                    state, target = experience_queue.get(timeout=5)
                    update_event.clear()
                    time.sleep(0.05)
                    global_network.net.fit(state, target, batch_size=1, verbose=0)
                    update_event.set()
                except queue.Empty:
                    print("Queue empty, trying again")

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
                num_iterations, num_test_trials, run_index
            )


if __name__ == '__main__':
    run_training()


