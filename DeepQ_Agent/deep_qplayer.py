from hfo import *

class Deep_QPlayer:
    """
    Class used to run testing on qnetwork.
    These players do not learn from experience.
    """
    def __init__(self, qnetwork, port, num_episodes, num_teammates, num_opponents):
        self.qnetwork = qnetwork
        self.hfo_env = None
        self.port = port
        self.num_episodes = num_episodes
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents

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
            timestep = 0
            while status == IN_GAME:
                timestep += 1
                state = self.hfo_env.getState()
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

            if status == SERVER_DOWN:
                self.hfo_env.act(QUIT)
                break



