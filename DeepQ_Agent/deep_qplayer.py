from hfo import *
from deep_qnetwork import Deep_QNetwork

class Deep_QPlayer:
    """
    Class used to run testing on qnetwork.
    These players do not learn from experience.
    """
    def __init__(self, qnetwork, port, num_episodes):
        self.qnetwork = qnetwork
        self.hfo_env = None
        self.port = port
        self.num_episodes = num_episodes

    def get_reward(self, state):
        reward = 0

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

                if action is not None:
                    reward = self.get_reward(state)
                    if old_state is not None:
                        self.experience_batches.append((old_state, reward, state, False))

                old_state = state.copy()

            if action is not None and state is not None:
                reward = self.get_reward(status)
                self.experience_batches.append((old_state, reward, state, True))
                self.qnetwork.save()
                self.update_network()

            if status == SERVER_DOWN:
                self.hfo_env.act(QUIT)
                self.qnetwork.save()
                break



