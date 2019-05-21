from hfo import *
import numpy as np
from deep_qnetwork import Global_QNetwork, Local_QNetwork
import state_representer

class Deep_QPlayer:
    """
    Class used to run testing on qnetwork.
    These players do not learn from experience.
    """
    def __init__(self, global_main_network, port, num_episodes,
                 num_teammates, num_opponents):
        self.global_main_network = global_main_network  # type: Global_QNetwork
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents

        self.local_network = None  # type: Local_QNetwork
        self.initialize_local_network()

        self.hfo_env = None
        self.port = port
        self.num_episodes = num_episodes

    def initialize_local_network(self):
            main_net_architecture = self.global_main_network.net.to_json()
            main_net_weights = self.global_main_network.net.get_weights()
            self.local_network = Local_QNetwork(
                main_net_architecture, main_net_weights,
                num_teammates=self.num_teammates
            )

    def connect(self):
        hfo_env = HFOEnvironment()
        hfo_env.connectToServer(feature_set=LOW_LEVEL_FEATURE_SET, server_port=self.port)
        self.hfo_env = hfo_env

    def run_episodes(self, output_file):
        if not self.hfo_env:
            raise(ValueError, "HFO Environment not detected, must run connect before run_episodes.")

        with open(output_file, 'w+') as out_file:
            for episode in range(0, self.num_episodes):
                status = IN_GAME
                history = []
                timestep = 0
                while status == IN_GAME:
                    timestep += 1
                    state = np.array(self.hfo_env.getState())
                    shaped_state = state.reshape((1,-1))

                    if int(state[12]) != 1:
                        history.append(state[0])
                        if len(history) > 5:
                            history.pop(0)

                        if len(history) == 5 and history[0] == history[4]:
                            self.hfo_env.act(REORIENT)
                            history = []
                            continue

                        self.hfo_env.act(MOVE)

                    else:
                        action, meme = self.local_network.get_action(shaped_state)

                        if action == 0:
                            print("DRIBBLE_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(DRIBBLE)
                        elif action == 1:
                            print("SHOOT_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(SHOOT)
                        elif self.num_teammates > 0:
                            print("PASS CHOSEN", flush=True, file=out_file)
                            teammate_number = round(state[
                                58 + (8 * self.num_teammates) + (8 * self.num_opponents)
                                + (action - 2)
                            ] * 100)
                            if teammate_number == -100: # can't pass to a teammate
                                self.hfo_env.act(DRIBBLE)
                            else:
                                self.hfo_env.act(PASS, teammate_number)

                    status = self.hfo_env.step()


                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break


class HL_Deep_QPlayer(Deep_QPlayer):
    def connect(self):
        hfo_env = HFOEnvironment()
        hfo_env.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=self.port)
        self.hfo_env = hfo_env

    def run_episodes(self, output_file):
        if not self.hfo_env:
            raise (ValueError, "HFO Environment not detected, must run connect before run_episodes.")

        with open(output_file, 'w+') as out_file:
            for episode in range(0, self.num_episodes):
                status = IN_GAME
                history = []
                timestep = 0
                while status == IN_GAME:
                    timestep += 1
                    state = np.array(self.hfo_env.getState())
                    shaped_state = state.reshape((1, -1))

                    if int(state[5]) != 1:
                        history.append((state[0], state[1]))
                        if len(history) > 5:
                            history.pop(0)

                        if len(history) == 5:
                            if history[0][0] == history[4][0] and history[0][1] == history[4][1]:
                                self.hfo_env.act(REORIENT)
                                history = []
                                continue

                        self.hfo_env.act(MOVE)

                    else:
                        action, meme = self.local_network.get_action(shaped_state)

                        if action == 0:
                            print("DRIBBLE_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(DRIBBLE)
                        elif action == 1:
                            print("SHOOT_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(SHOOT)
                        elif self.num_teammates > 0:
                            print("PASS CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(PASS, state[15 + 6 * (action - 2)])

                    status = self.hfo_env.step()

                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break


class Discrete_Deep_QPlayer(Deep_QPlayer):
    def connect(self):
        hfo_env = HFOEnvironment()
        hfo_env.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=self.port)
        self.hfo_env = hfo_env

    def run_episodes(self, output_file):
        if not self.hfo_env:
            raise (ValueError, "HFO Environment not detected, must run connect before run_episodes.")

        with open(output_file, 'w+') as out_file:
            for episode in range(0, self.num_episodes):
                status = IN_GAME
                history = []
                timestep = 0
                while status == IN_GAME:
                    timestep += 1
                    features = np.array(self.hfo_env.getState())
                    state = np.array(
                        state_representer.get_representation(features, self.num_teammates)
                    )
                    shaped_state = state.reshape((1, -1))

                    if int(features[5]) != 1:
                        history.append((features[0], features[1]))
                        if len(history) > 5:
                            history.pop(0)

                        if len(history) == 5:
                            if history[0][0] == history[4][0] and history[0][1] == history[4][1]:
                                self.hfo_env.act(REORIENT)
                                history = []
                                continue

                        self.hfo_env.act(MOVE)

                    else:
                        action, meme = self.local_network.get_action(shaped_state)

                        if action == 0:
                            print("DRIBBLE_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(DRIBBLE)
                        elif action == 1:
                            print("SHOOT_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(SHOOT)
                        elif self.num_teammates > 0:
                            print("PASS CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(PASS, features[15 + 6 * (action - 2)])

                    status = self.hfo_env.step()

                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break