import click
import subprocess
import os
import time
from tqdm import trange
from util import clean_keep_lines
from deep_qnetwork import Global_QNetwork
from deep_qplayer import Deep_QPlayer
from agent_thread import Player_Thread

def run_testing(deep_network_path, output_directory, port=6000, num_agents=2,
                num_opponents=2, num_iterations=0, num_test_trials=0, run_index=0):
    num_teammates = num_agents - 1
    for iteration in trange(num_iterations):
        network_path = os.path.join(deep_network_path, "iter_" + str(iteration), 'main_net.h5')
    
        global_network = Global_QNetwork(load_location=network_path)

        output_file = os.path.join(output_directory, 'test_iter_' + str(iteration) + '.txt')
        with open(output_file, 'w+') as outfile:
            hfo_process = subprocess.Popen(args=['./bin/HFO',
                                                    '--offense-agents='+str(num_agents),
                                                    '--defense-npcs='+str(num_opponents),
                                                    '--offense-on-ball', '1',
                                                    '--trials', str(num_test_trials),
                                                    '--port', str(port),
                                                    '--headless'],
                                            stdout=outfile)

        time.sleep(15)

        deep_players = []
        for _ in range(num_agents):
            deep_learner = Deep_QPlayer(
                global_network, port, num_test_trials, num_teammates, num_opponents
            )
            deep_players.append(deep_learner)
            time.sleep(10)

        player_threads = []
        for learner_index in range(0, len(deep_players)):
            print("Test player " + str(learner_index) + " started.")
            player = Player_Thread(
                deep_players[learner_index], os.devnull
            )
            player_threads.append(player)
            player.start()
            time.sleep(5)

        hfo_process.wait()

        clean_keep_lines(output_iteration_dir, [output_file], 20)