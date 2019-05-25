import numpy as np
from hfo import *
import state_representer


class QPlayer:
    def __init__(self, num_states, num_actions, num_teammates, num_opponents, port, num_episodes, q_table_in=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        if q_table_in:
            self.load(q_table_in)
        else:
            self.q_table = np.zeros((num_states, num_actions))
        self.old_state = None

        self.hfo_env = None
        self.port = port
        self.num_episodes = num_episodes

    def set_invalid(self, state, invalid_teammates):
        """
        Dangerous method! Should not be used for updating q values,
        only used to set value of passing to undetected teammate
        to reward of -inf.

        :param state:
        :param invalid_teammates:
        :return:
        """
        # if invalid actions have already been set do nothing.
        if not -np.inf in self.q_table[state]:
            for index in range(len(invalid_teammates)):
                if invalid_teammates[index] == 0: # teammate invalid
                    self.q_table[state][2 + index] = -np.inf

    def get_action(self, state):
        max_list = np.where(self.q_table[state] == self.q_table[state].max())
        if len(max_list[0]) > 1:
            action = np.random.randint(0, len(max_list[0]))
            return action

        return np.argmax(self.q_table[state])

    def connect(self):
        hfo_env = HFOEnvironment()
        hfo_env.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=self.port)
        self.hfo_env = hfo_env

    def run_episodes(self, output_file):
        if not self.hfo_env:
            raise(ValueError, "HFO Environment not detected, must call 'connect' before calling 'run_episodes'")

        with open(output_file, 'w+') as out_file:
            for episode in range(0, self.num_episodes):
                status = IN_GAME
                history = []
                while status == IN_GAME:
                    features = self.hfo_env.getState()
                    if int(features[5]) != 1:
                        history.append((features[0], features[1]))
                        if len(history) > 5:
                            history.pop(0)

                        # ensures agent does not get stuck for prolonged periods
                        if len(history) == 5:
                            if history[0][0] == history[4][0] and history[0][1] == history[4][1]:
                                self.hfo_env.act(REORIENT)
                                history = []
                                continue

                        self.hfo_env.act(MOVE)
                    else:
                        state, valid_teammates = state_representer.get_representation(features, self.num_teammates)

                        action = self.get_action(state)

                        if action == 0:
                            print("Action Taken: DRIBBLE \n")
                            self.hfo_env.act(DRIBBLE)
                        elif action == 1:
                            print("Action Taken: SHOOT \n")
                            self.hfo_env.act(SHOOT)
                        elif action > 1:
                            print("Action Taken: PASS -> {0} \n".format(action - 2))
                            self.hfo_env.act(PASS, features[15 + 6 * (action - 2)])
                    status = self.hfo_env.step()

                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    break

    def load(self, q_table_in):
        try:
            self.q_table = np.load(q_table_in)
        except (IOError, OSError) as ioe:
            print("Error with file: " + q_table_in)
            print(ioe.strerror)
            print("Reinitializing Q Table")
            self.q_table = np.zeros((self.num_states, self.num_actions))


