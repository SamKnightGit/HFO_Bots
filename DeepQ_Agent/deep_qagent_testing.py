import click
import subprocess
import os
import time
from tqdm import trange
from util import clean_keep_lines, clean_dir_keep_lines
from deep_qnetwork import Global_QNetwork
from deep_qplayer import Deep_QPlayer, HL_Deep_QPlayer, Discrete_Deep_QPlayer
from agent_thread import Player_Thread

def run_testing(deep_network_path, output_directory, port=6000, num_agents=2,
                num_opponents=2, num_iterations=0, num_test_trials=0, state_space='d'):
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
                                                    '--headless',
                                                    '--no-logging'],
                                            stdout=outfile)

        time.sleep(15)

        deep_players = []
        for _ in range(num_agents):
            if state_space == 'hl':
                deep_player = HL_Deep_QPlayer(
                    global_network, port, num_test_trials, num_teammates, num_opponents
                )
            elif state_space == 'll':
                deep_player = Deep_QPlayer(
                    global_network, port, num_test_trials, num_teammates, num_opponents
                )
            else:
                deep_player = Discrete_Deep_QPlayer(
                    global_network, port, num_test_trials, num_teammates, num_opponents
                )
            deep_players.append(deep_player)
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

        clean_keep_lines(output_directory, [output_file], 20)

# if __name__ == "__main__":
#     run_testing('/home/samdknight/Documents/Edinburgh/4th/Dissertation/HFO/example/custom_agents/HFO_Bots/DeepQ_Agent/nn_models/2019-02-26T01:22:50.743288_agents2_opponents1_eps1.0_lr0.9_pg5/run_0',
#                 '/home/samdknight/Documents/Edinburgh/4th/Dissertation/HFO/example/custom_agents/HFO_Bots/DeepQ_Agent/output/2019-02-26T01:22:50.743288_agents2_opponents1_eps1.0_lr0.9_pg5/run_0',
#                 port=7500, num_opponents=1, num_iterations=10, num_test_trials=1000, high_level=True)
#
#
#     output_dir = os.path.join(
#         '/home/samdknight/Documents/Edinburgh/4th/Dissertation/HFO/example/custom_agents/HFO_Bots/DeepQ_Agent/output/2019-02-21T19:15:27.171412_agents2_opponents1_eps0.8_lr0.1',
#         'run_0'
#     )
#     for train_dir in os.listdir(output_dir):
#         train_dir = os.path.join(output_dir, train_dir)
#         if os.path.isdir(train_dir):
#             clean_dir_keep_lines(train_dir, 20)