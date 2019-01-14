from hfo import *
from deep_qnetwork import Deep_QNetwork

class Deep_QLearner:
    def __init__(self, qnetwork, port, num_episodes, num_teammates, num_opponents):
        self.qnetwork = qnetwork # type: Deep_QNetwork
        self.hfo_env = None
        self.port = port
        self.num_episodes = num_episodes
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.experience_batches = []

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

    def run_episodes(self):
        if not self.hfo_env:
            raise(ValueError, "HFO Environment not detected, must run connect before run_episodes.")

        for episode in range(0, self.num_episodes):
            status = IN_GAME
            action = None
            old_state = None
            state = None
            timestep = 0
            while status == IN_GAME:
                timestep += 1
                state = self.hfo_env.getState()

                if int(state[5]) != 1:
                    self.hfo_env.act(MOVE)
                else:
                    if action is not None:
                        reward = self.get_reward(state)
                        if old_state is not None:
                            self.experience_batches.append((old_state, reward, state, False))

                    action = self.qnetwork.get_action(state)

                    if action == 0:
                        self.hfo_env.act(DRIBBLE)
                    elif action == 1:
                        self.hfo_env.act(SHOOT)
                    elif self.num_teammates > 0:
                        self.hfo_env.act(PASS, state[
                            57 + (8 * self.num_teammates) + (8 * self.num_opponents)
                            + (action - 2)
                        ])
                old_state = state.copy()
                status = self.hfo_env.step()

            if action is not None and state is not None:
                reward = self.get_reward(status)
                self.experience_batches.append((old_state, reward, state, True))
                self.qnetwork.save_network()
                self.update_network()

            if status == SERVER_DOWN:
                self.hfo_env.act(QUIT)
                self.qnetwork.save_network()
                break

    def update_network(self):
        self.qnetwork.update_main_net(self.experience_batches)
        self.experience_batches = []


