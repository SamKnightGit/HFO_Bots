from hfo import *
from threading import Event
from deep_qnetwork import Global_QNetwork, Learning_QNetwork, Learning_DoubleQNetwork
from typing import List
import state_representer
import reward_functions



class Deep_QLearner:
    def __init__(self, global_main_network, reward_function_name, experience_list, port, double_q,
                 learning_rate, epsilon, num_episodes, num_teammates, num_opponents):
        self.global_main_network = global_main_network  # type: Global_QNetwork
        self.reward_function_name = reward_function_name
        self.reward_function = self.get_reward_function(reward_function_name)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.double_q = double_q
        self.local_network = None  # type: Learning_QNetwork
        self.initialize_local_network()

        self.shared_experience_list = experience_list  # type: List
        self.hfo_env = None
        self.port = port
        self.num_episodes = num_episodes
        self.time_until_target_update = 15
        # currently unused -- updated after every episode
        # self.time_until_main_update = 10

    def initialize_local_network(self):
        main_net_architecture = self.global_main_network.net.to_json()
        main_net_weights = self.global_main_network.net.get_weights()
        if self.double_q:
            self.local_network = Learning_DoubleQNetwork(
                main_net_architecture, main_net_weights, learning_rate=self.learning_rate,
                epsilon=self.epsilon, num_teammates=self.num_teammates
            )
        else:
            self.local_network = Learning_QNetwork(
                main_net_architecture, main_net_weights, learning_rate=self.learning_rate,
                epsilon=self.epsilon, num_teammates=self.num_teammates
            )

    def update_local_main_network(self):
        weights = self.global_main_network.net.get_weights()
        self.local_network.update_main_network(weights)

    def update_local_target_network(self):
        weights = self.global_main_network.net.get_weights()
        self.local_network.update_target_network(weights)

    def get_reward_function(self, reward_function_name):
        if reward_function_name == 'sparse':
            return reward_functions.get_sparse_reward
        elif reward_function_name == 'simple':
            return reward_functions.ll_simple_reward
        else: #advanced
            return reward_functions.ll_advanced_reward


    def connect(self):
        hfo_env = HFOEnvironment()
        hfo_env.connectToServer(feature_set=LOW_LEVEL_FEATURE_SET, server_port=self.port)
        self.hfo_env = hfo_env

    def run_episodes(self, output_file):
        if not self.hfo_env:
            raise(ValueError, "HFO Environment not detected, must run connect before run_episodes.")

        with open(output_file, 'w+') as out_file:
            for episode in range(0, self.num_episodes):
                if episode % self.time_until_target_update == 0:
                    self.update_local_target_network()
                status = IN_GAME
                action = None
                old_state = None
                state = None
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
                        if action is not None and old_state is not None:
                            if self.reward_function_name == 'sparse':
                                reward = self.reward_function(status)
                            else:
                                reward = self.reward_function(old_state, action, state)
                            target_val = self.local_network.get_target((old_state, action, reward, shaped_state, False))
                            self.shared_experience_list.append((old_state, target_val))

                        action, qvalue_arr = self.local_network.get_action(shaped_state)
                        print("Qval array: " + str(qvalue_arr), flush=True, file=out_file)

                        if action == 0:
                            self.hfo_env.act(DRIBBLE)
                        elif action == 1:
                            self.hfo_env.act(SHOOT)
                        elif self.num_teammates > 0:
                            teammate_number = round(state[
                                58 + (8 * self.num_teammates) + (8 * self.num_opponents)
                                + (action - 2)
                            ] * 100)
                            if teammate_number == -100: # can't pass to a teammate
                                self.hfo_env.act(DRIBBLE)
                            else:
                                self.hfo_env.act(PASS, teammate_number)

                    old_state = np.copy(shaped_state)
                    status = self.hfo_env.step()

                if action is not None and state is not None:
                    shaped_state = state.reshape((1, -1))
                    if self.reward_function_name == 'sparse':
                        reward = self.reward_function(status)
                    else:
                        reward = self.reward_function(old_state, action, state)
                    if action == 0:
                        print("DRIBBLE_CHOSEN with reward " + str(reward), flush=True, file=out_file)
                    elif action == 1:
                        print("SHOOT_CHOSEN with reward " + str(reward), flush=True, file=out_file)
                    else:
                        print("PASS_CHOSEN with reward " + str(reward), flush=True, file=out_file)

                    target_val = self.local_network.get_target((old_state, action, reward, shaped_state, True))
                    self.shared_experience_list.append((old_state, target_val))
                    self.update_local_main_network()

                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break


class HL_Deep_QLearner(Deep_QLearner):
    def get_reward_function(self, reward_function_name):
        if reward_function_name == 'sparse':
            return reward_functions.get_sparse_reward
        elif reward_function_name == 'simple':
            return reward_functions.hl_simple_reward
        else: #advanced
            return reward_functions.hl_advanced_reward

    def connect(self):
        hfo_env = HFOEnvironment()
        hfo_env.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=self.port)
        self.hfo_env = hfo_env

    def run_episodes(self, output_file):
        if not self.hfo_env:
            raise(ValueError, "HFO Environment not detected, must run connect before run_episodes.")

        with open(output_file, 'w+') as out_file:
            for episode in range(0, self.num_episodes):
                if episode % self.time_until_target_update == 0:
                    self.update_local_target_network()
                status = IN_GAME
                action = None
                old_state = None
                state = None
                history = []
                timestep = 0
                while status == IN_GAME:
                    timestep += 1
                    state = np.array(self.hfo_env.getState())
                    shaped_state = state.reshape((1,-1))

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
                        if action is not None and old_state is not None:
                            if self.reward_function_name == 'sparse':
                                reward = self.reward_function(status)
                            else:
                                reward = self.reward_function(old_state, action, shaped_state)
                            target_val = self.local_network.get_target((old_state, action, reward, shaped_state, False))
                            self.shared_experience_list.append((old_state, target_val))

                        action, qvalue_arr = self.local_network.get_action(shaped_state)
                        print("Qval array: " + str(qvalue_arr), flush=True, file=out_file)

                        if action == 0:
                            print("DRIBBLE_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(DRIBBLE)
                        elif action == 1:
                            print("SHOOT_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(SHOOT)
                        elif self.num_teammates > 0:
                            print("PASS CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(PASS, state[15 + 6 * (action-2)])

                    old_state = np.copy(shaped_state)
                    status = self.hfo_env.step()

                if action is not None and state is not None:
                    shaped_state = state.reshape((1, -1))
                    if self.reward_function_name == 'sparse':
                        reward = self.reward_function(status)
                    else:
                        reward = self.reward_function(old_state, action, shaped_state)
                    target_val = self.local_network.get_target((old_state, action, reward, shaped_state, True))
                    self.shared_experience_list.append((old_state, target_val))
                    self.update_local_main_network()

                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break


class Discrete_Deep_QLearner(Deep_QLearner):
    def get_reward_function(self, reward_function_name):
        if reward_function_name == 'sparse':
            return reward_functions.get_sparse_reward
        elif reward_function_name == 'simple':
            return reward_functions.discrete_simple_reward
        else: #advanced
            return reward_functions.discrete_advanced_reward

    def connect(self):
        hfo_env = HFOEnvironment()
        hfo_env.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=self.port)
        self.hfo_env = hfo_env

    def run_episodes(self, output_file):
        if not self.hfo_env:
            raise(ValueError, "HFO Environment not detected, must run connect before run_episodes.")

        with open(output_file, 'w+') as out_file:
            for episode in range(0, self.num_episodes):
                status = IN_GAME
                action = None
                old_state = None
                state = None
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
                        if action is not None and old_state is not None:
                            if self.reward_function_name == 'sparse':
                                reward = self.reward_function(status)
                            else:
                                reward = self.reward_function(old_state, action, shaped_state)
                                print(reward)
                            target_val = self.local_network.get_target((old_state, action, reward, shaped_state, False))
                            self.shared_experience_list.append((old_state, target_val))

                        action, qvalue_arr = self.local_network.get_action(shaped_state)
                        print("Qval array: " + str(qvalue_arr), flush=True, file=out_file)

                        if action == 0:
                            print("DRIBBLE_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(DRIBBLE)
                        elif action == 1:
                            print("SHOOT_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(SHOOT)
                        elif self.num_teammates > 0:
                            print("PASS CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(PASS, features[15 + 6 * (action-2)])

                    old_state = np.copy(shaped_state)
                    status = self.hfo_env.step()

                if action is not None and state is not None:
                    shaped_state = state.reshape((1, -1))
                    if self.reward_function_name == 'sparse':
                        reward = self.reward_function(status)
                    else:
                        reward = self.reward_function(old_state, action, shaped_state)
                    target_val = self.local_network.get_target((old_state, action, reward, shaped_state, True))
                    self.shared_experience_list.append((old_state, target_val))
                    self.update_local_main_network()

                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break
