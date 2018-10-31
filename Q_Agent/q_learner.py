import numpy as np
import random


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
            self.q_table[self.old_state, action] = \
                (1 - self.learn_rate) * self.q_table[self.old_state, action] + \
                self.learn_rate * (reward + self.discount * np.amax(self.q_table[state]))

        self.old_state = state

    def get_action(self, state):
        explore = True if random.random() <= self.eps else False
        if explore:
            return random.randint(0, self.num_actions)

        return np.argmax(self.q_table[state])

    def adjust_epsilon(self, timestep):
        """
        Adjust epsilon used for decreasing exploration after agent has been sufficiently trained

        :param int timestep: Timestep of game
        """
        # TODO: Find a more suitable function for annealing epsilon
        self.eps = 1 / 1 + timestep

    def save(self):
        np.save(self.table_out, self.q_table)

