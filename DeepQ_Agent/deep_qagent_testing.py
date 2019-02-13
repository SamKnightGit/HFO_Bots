import click
import subprocess
import os
import time
from tqdm import trange
from deep_qnetwork import Global_QNetwork
from deep_qplayer import Deep_QPlayer
from agent_thread import Player_Thread

# @click.command()
# @click.option('--port', '-p', default=6000)
# @click.option('--deep_network_path', '-dn')
# @click.option('--output_directory', '-o',
#               help="Path to directory where output files will be stored.")
# @click.option('--num_agents', '-tm', default=2)
# @click.option('--num_opponents', '-op', default=2)
# @click.option('--num_iterations', '-n', default=0,
#               help="The number of training iterations to run the bots on.")
# @click.option('--num_test_iterations', '-tn', default=0,
#               help="Number of test iterations to run for each train iteration.")
# @click.option('--num_test_trials', '-tt', default=0,
#               help="Number of test trials to run for each test iteration."
#                    "Defaults to number train iterations when not set.")
def run_testing(deep_network_path, output_directory, port=6000, num_agents=2,
                num_opponents=2, num_iterations=0, num_test_iterations=0, num_test_trials=0):
    num_teammates = num_agents - 1
    for iteration in trange(num_iterations):
        output_iteration_dir = os.path.join(output_directory, 'test_iter_' + str(iteration))
        os.makedirs(output_iteration_dir, exist_ok=True)
        # Dummy variables as network is being loaded.
        network_path = os.path.join(deep_network_path, "iter_" + str(iteration), 'main_net.h5')
        global_network = Global_QNetwork(0, 0, 0,
                                       load_location=network_path)
        for test_iteration in trange(num_test_iterations):
            output_file = os.path.join(
                output_iteration_dir,
                'test_run_' + str(test_iteration) + '.txt'
            )
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


def clean_keep_lines(logging_directory, logs, num_lines):

    temppath = os.path.join(logging_directory, 'tempfile.txt')

    for log_path in logs:
        with open(temppath, 'w+') as tempfile:
            subprocess.run(['tail', '-n', str(num_lines), log_path], stdout=tempfile)
        subprocess.run(['mv', temppath, log_path])