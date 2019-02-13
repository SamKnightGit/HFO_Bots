from hfo import *
from deep_qnetwork import Global_QNetwork, Learning_QNetwork
import queue


class Deep_QLearner:
    def __init__(self, global_main_network, experience_queue, port, learning_rate,
                 epsilon, num_episodes, num_teammates, num_opponents):
        self.global_main_network = global_main_network  # type: Global_QNetwork
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents

        self.local_network = None  # type: Learning_QNetwork
        self.initialize_local_network()

        self.shared_experience_queue = experience_queue  # type: queue.Queue
        self.hfo_env = None
        self.port = port
        self.num_episodes = num_episodes
        self.time_until_target_update = 25
        self.time_until_main_update = 5

    def initialize_local_network(self):
            main_net_architecture = self.global_main_network.net.to_json()
            main_net_weights = self.global_main_network.net.get_weights()
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

    def get_reward(self, state):
        reward = 0  # type: int

        if state == GOAL:
            reward = 1

        elif state == CAPTURED_BY_DEFENSE:
            reward = -1

        elif state == OUT_OF_BOUNDS:
            reward = -1

        elif state == OUT_OF_TIME:
            reward = -1

        return reward

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
                            reward = self.get_reward(status)
                            input_state, target_val = self.local_network.get_target((old_state, reward, shaped_state, False))
                            self.shared_experience_queue.put((input_state, target_val))

                        self.update_local_main_network()
                        action = self.local_network.get_action(shaped_state)

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
                            self.hfo_env.act(PASS, teammate_number)

                    if (timestep % self.time_until_target_update) == 0:
                        self.update_local_target_network()

                    if (timestep % self.time_until_main_update) == 0:
                        self.update_local_main_network()

                    old_state = np.copy(shaped_state)
                    status = self.hfo_env.step()

                if action is not None and state is not None:
                    shaped_state = state.reshape((1, -1))
                    reward = self.get_reward(status)
                    input_state, target_val = self.local_network.get_target((old_state, reward, shaped_state, True))
                    self.shared_experience_queue.put((input_state, target_val))


                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break


class Discrete_Deep_QLearner(Deep_QLearner):
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
                            reward = self.get_reward(status)
                            input_state, target_val = self.local_network.get_target((old_state, reward, shaped_state, False))
                            self.shared_experience_queue.put((input_state, target_val))

                        self.update_local_main_network()
                        action = self.local_network.get_action(shaped_state)

                        if action == 0:
                            print("DRIBBLE_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(DRIBBLE)
                        elif action == 1:
                            print("SHOOT_CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(SHOOT)
                        elif self.num_teammates > 0:
                            print("PASS CHOSEN", flush=True, file=out_file)
                            self.hfo_env.act(PASS, state[15 + 6 * (action-2)])

                    if (timestep % self.time_until_target_update) == 0:
                        self.update_local_target_network()

                    if (timestep % self.time_until_main_update) == 0:
                        self.update_local_main_network()

                    old_state = np.copy(shaped_state)
                    status = self.hfo_env.step()

                if action is not None and state is not None:
                    shaped_state = state.reshape((1, -1))
                    reward = self.get_reward(status)
                    input_state, target_val = self.local_network.get_target((old_state, reward, shaped_state, True))
                    self.shared_experience_queue.put((input_state, target_val))

                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break

