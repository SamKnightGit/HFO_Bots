import click
import subprocess
import os
import time
from deep_qnetwork import Global_QNetwork
from deep_qplayer import Deep_QPlayer, HL_Deep_QPlayer, Discrete_Deep_QPlayer
from agent_thread import Player_Thread


@click.command()
@click.option('--deep_network_path', '-n')
@click.option('--port', '-p', default=6000)
@click.option('--num_agents', '-tm', default=2)
@click.option('--num_opponents', '-op', default=2)
@click.option('--iteration_num', '-i', default=1)
@click.option('--num_test_trials', '-tt', default=5)
@click.option('--state_space', '-ss', default='d',
              type=click.Choice(['d', 'hl', 'll']))
def run_testing(deep_network_path, port, num_agents, num_opponents,
                iteration_num, num_test_trials, state_space):
    num_teammates = num_agents - 1
    network_path = os.path.join(deep_network_path, "iter_" + str(iteration_num), 'main_net.h5')

    global_network = Global_QNetwork(load_location=network_path)

    hfo_process = subprocess.Popen(args=['./bin/HFO',
                                         '--offense-agents=' + str(num_agents),
                                         '--defense-npcs=' + str(num_opponents),
                                         '--offense-on-ball', '1',
                                         '--trials', str(num_test_trials),
                                         '--port', str(port),
                                         '--no-sync',
                                         '--no-logging'])

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


if __name__ == "__main__":
    run_testing()