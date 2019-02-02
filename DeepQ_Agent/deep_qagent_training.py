#!/usr/bin/env python

import subprocess
import os
import queue
from typing import List

import click
import random
import time
from datetime import datetime
from tqdm import trange
from deep_qnetwork import Global_QNetwork, Local_QNetwork
from deep_qlearner import Deep_QLearner
from agent_thread import Agent_Thread

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
@click.option('--learning_rate', '-lr', default=0.10)
@click.option('--num_agents', '-tm', default=2)
@click.option('--num_opponents', '-op', default=2)
@click.option('--trials_per_iteration', '-t', default=1)
@click.option('--num_iterations', '-n', default=0)
@click.option('--num_parallel_games', '-pg', default=5)
@click.option('--save_network_dir', '-ns', default=base_network_dir)
@click.option('--load_network_dir', '-nl', default=base_network_dir)
@click.option('--logging_dir', '-l', default=base_logging_dir)
@click.option('--output_dir', '-o', default=base_output_dir)
def run_training(port, seed, learning_rate, num_agents, num_opponents, trials_per_iteration,
                 num_iterations, num_parallel_games, save_network_dir, load_network_dir,
                 logging_dir, output_dir):
    experience_queue = queue.Queue()

    num_teammates = num_agents - 1
    state_dimensions = 59 + (9 * num_teammates) + (9 * num_opponents)

    global_network = Global_QNetwork(
        state_dimensions, 0.1, num_teammates
    )

    for iteration in trange(int(num_iterations)):

        save_net_dir = os.path.join(save_network_dir, 'iter_' + str(iteration))
        os.makedirs(save_net_dir, exist_ok=True)

        global_network.set_save_location(save_net_dir)

        log_dir = os.path.join(logging_dir, 'iter_' + str(iteration))
        os.makedirs(log_dir, exist_ok=True)

        out_dir = os.path.join(output_dir, 'iter_' + str(iteration))
        os.makedirs(out_dir, exist_ok=True)

        hfo_processes = []
        learners = []
        for game_index in range(0, num_parallel_games):
            unique_port = port + 5 * game_index # consecutive ports does not work well
            unique_seed = seed + game_index

            output_file = os.path.join(out_dir, 'game_' + str(game_index) + '.txt')
            with open(output_file, 'w+') as outfile:
                hfo_game = subprocess.Popen(args=['./bin/HFO', '--offense-agents='+str(num_agents), '--defense-npcs=2',
                                                 '--offense-on-ball', '1', '--trials', str(trials_per_iteration),
                                                 '--port', str(unique_port), '--seed', str(unique_seed), '--headless'],
                                            stdout=outfile,
                                            stderr=outfile)

            hfo_processes.append(hfo_game)

        time.sleep(15)
        deep_learners = []
        for game_index in range(0, num_parallel_games):
            print("Connecting for game: " + str(game_index))
            unique_port = port + 5 * game_index

            for agent_index in range(0, int(num_agents)):
                deep_learner = Deep_QLearner(
                    global_network, experience_queue, unique_port, learning_rate,
                    state_dimensions, trials_per_iteration, num_teammates, num_opponents
                )
                print("Agent " + str(agent_index) +
                      " connected for game: " + str(game_index))
                deep_learners.append(deep_learner)

        finished_list = [0] * len(deep_learners)  # type: List[int]
        for learner_index in range(0, len(deep_learners)):
            print("Learner " + str(learner_index) + " started.")
            log_file = os.path.join(log_dir, 'learner_' + str(learner_index) + '.txt')
            learner_thread = Agent_Thread(deep_learners[learner_index], learner_index, finished_list,
                                          log_file)
            learners.append(learner_thread)
            learner_thread.start()
            time.sleep(5)

        while 0 in finished_list:
            try:
                state, target = experience_queue.get(timeout=5)
                global_network.net.fit(state, target, batch_size=1, verbose=0)
            except queue.Empty as qe:
                print("Queue empty, trying again")

        for learner in learners:
            learner.join()

        # Wait for all servers to close
        for hfo_process in hfo_processes:
            hfo_process.wait()

        global_network.save_network()


if __name__ == '__main__':
    run_training()