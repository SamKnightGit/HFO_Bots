from hfo import *
import subprocess
import _thread
import click
import random
import time
from datetime import datetime
from tqdm import tqdm
from .deep_qnetwork import Deep_QNetwork
from .deep_qlearner import Deep_QLearner

creation_time = datetime.isoformat(datetime.today())

network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn_models')
network_dir = os.path.join(network_dir, creation_time)

output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
output_dir = os.path.join(output_dir, creation_time)

DEEP_QNETWORK_LOAD_PATH = ''
DEEP_QNETWORK_SAVE_PATH = ''


@click.command()
@click.option('--port', '-p', default=6000)
@click.option('--seed', '-s', default=random.randint(1,5000))
@click.option('--deep_network_path', '-dn', default=DEEP_QNETWORK_LOAD_PATH)
@click.option('--num_teammates', '-tm', default=2)
@click.option('--num_opponents', '-op', default=2)
@click.option('--trials_per_iteration', '-t', default=1)
@click.option('--num_iterations', '-n', default=0)
@click.option('--num_parallel_games', '-pg', default=5)
def run_training(port, seed, deep_network_path, num_teammates, num_opponents,
                 trials_per_iteration, num_iterations, num_parallel_games):

    STATE_DIMENSIONS = 59 + (9 * num_teammates) + (9 * num_opponents)

    deep_network = Deep_QNetwork(STATE_DIMENSIONS,
                                 load_location=deep_network_path,
                                 save_location=deep_network_path,
                                 num_teammates=num_teammates,
                                 num_opponents=num_opponents)

    for iteration in tqdm(0, num_iterations):
        hfo_processes = []

        for game_index in range(0, num_parallel_games):
            unique_port = port + 5 * game_index # consecutive ports does not work well
            unique_seed = seed + game_index
            hfo_game = subprocess.Popen(args=['./bin/HFO', '--offense-agents=2', '--defense-npcs=2',
                                                 '--offense-on-ball', '1', '--trials', trials_per_iteration,
                                                 '--port', unique_port, '--seed', unique_seed, '--headless'])

            hfo_processes.append(hfo_game)


        time.sleep(15)

        for game_index in range(0, num_parallel_games):
            unique_port = port + 5 * game_index

            deep_learners = []

            for _ in range(0, num_teammates):
                deep_learner = Deep_QLearner(deep_network, unique_port, trials_per_iteration, num_teammates)
                deep_learner.connect()
                deep_learners.append(deep_learner)
                time.sleep(5)

            for learner in deep_learners:
                _thread.start_new_thread(learner.run_episodes(), ())

        for hfo_process in hfo_processes:
            hfo_process.wait()




if __name__ == '__main__':
    run_training()