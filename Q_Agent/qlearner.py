import numpy as np
from hfo import *
from util.helpers import get_reward, reward_printer
import state_representer


class QLearner:
    def __init__(self, num_states, num_actions, num_teammates, num_opponents, port, num_episodes,
                 epsilon=0.10, learning_rate=0.10, discount_factor=0.9, q_table_in=None, q_table_out=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.epsilon = epsilon
        self.learn_rate = learning_rate
        self.discount = discount_factor
        self.table_out = q_table_out
        if q_table_in:
            self.load(q_table_in)
        else:
            self.q_table = np.zeros((num_states, num_actions))
        self.old_state = None

        self.hfo_env = None
        self.port = port
        self.num_episodes = num_episodes

    def update(self, state, action, reward):
        if self.old_state:
            self.q_table[self.old_state][action] *= (1 - self.learn_rate)
            self.q_table[self.old_state][action] += self.learn_rate * (reward + self.discount * np.amax(self.q_table[state]))
        self.old_state = state

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

    def get_action(self, state, valid_teammates):
        # Assemble valid actions range
        valid_actions = self.num_actions - valid_teammates.count(0)

        explore = True if np.random.random() < self.epsilon else False
        if explore:
            random_action = np.random.randint(0, valid_actions)
            if random_action >= 2:
                teammate_chosen = random_action - 2
                passable_teammate = [index for index, teammate_valid
                                   in enumerate(valid_teammates)
                                   if teammate_valid == 1][teammate_chosen]
                random_action = 2 + passable_teammate
            return random_action

        # If multiple equal q-values, pick randomly
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
                action = None
                state = None
                history = []
                while status == IN_GAME:
                    features = self.hfo_env.getState()
                    # Print off features in a readable manner
                    # feature_printer(features, args.numTeammates, args.numOpponents)

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
                        state, valid_teammates = state_representer.get_representation(features, args.numTeammates)
                        print("Valid Teammates: ", valid_teammates)
                        if 0 in valid_teammates:
                            self.set_invalid(state, valid_teammates)

                        if action is not None:
                            reward = get_reward(status)
                            reward_printer(state, action, reward)
                            self.update(state, action, reward)

                        action = self.get_action(state, valid_teammates)

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

                if action is not None and state is not None:
                    reward = get_reward(status)
                    reward_printer(state, action, reward)
                    self.update(state, action, reward)
                    self.clear()
                    self.save()

                if status == SERVER_DOWN:
                    self.hfo_env.act(QUIT)
                    self.save()
                    break

            self.save()

    def clear(self):
        self.old_state = None

    def save(self):
        np.save(self.table_out, self.q_table)

    def load(self, q_table_in):
        try:
            self.q_table = np.load(q_table_in)
        except (IOError, OSError) as ioe:
            print("Error with file: " + q_table_in)
            print(ioe.strerror)
            print("Reinitializing Q Table")
            self.q_table = np.zeros((self.num_states, self.num_actions))


