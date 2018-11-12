import numpy as np


class QLearner:
    def __init__(self, num_states=0, num_actions=0, epsilon=0.10, learning_rate=0.1, discount_factor=0.9,  q_table_in=None, q_table_out=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = epsilon
        self.learn_rate = learning_rate
        self.discount = discount_factor
        self.table_out = q_table_out
        self.old_state = None
        if q_table_in:
            try:
                self.q_table = np.load(q_table_in)
            except IOError as ioe:
                print("Error with file: " + q_table_in)
                print(ioe.strerror)
                print("Reinitializing Q Table")
                self.q_table = np.zeros((num_states, num_actions))
        else:
            self.q_table = np.zeros((num_states, num_actions))

    def update(self, state, action, reward):
        if self.old_state:
            self.q_table[self.old_state, action] *= (1 - self.learn_rate) 
            self.q_table[self.old_state, action] += self.learn_rate * (reward + self.discount * np.amax(self.q_table[state]))
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
                if invalid_teammates[index] == 0:
                    self.q_table[state][2 + index] = -np.inf

    def get_action(self, state, valid_teammates):
        # Assemble valid actions range
        valid_actions = self.num_actions - valid_teammates.count(0)

        explore = True if np.random.random() <= self.eps else False
        if explore:
            random_action = np.random.randint(0, valid_actions)
            if random_action >= 2:
                teammate_chosen = random_action - 2
                passable_teammate = [index for index, teammate_valid
                                   in enumerate(valid_teammates)
                                   if teammate_valid == 1][teammate_chosen]
                random_action = 2 + passable_teammate
                print("PASS TO: ", random_action)
            return random_action

        return np.argmax(self.q_table[state])

    def adjust_epsilon(self, timestep):
        """
        Adjust epsilon used for decreasing exploration after agent has been sufficiently trained

        :param int timestep: Timestep of game
        """
        # TODO: Find a more suitable function for annealing epsilon
        self.eps = 1 / 1 + timestep

    def clear(self):
        self.old_state = None

    def save(self):
        np.save(self.table_out, self.q_table)

    def load(self, q_table_in):
        try:
            self.q_table = np.load(q_table_in)
        except IOError as ioe:
            print("Error with file: " + q_table_in)
            print(ioe.strerror)


