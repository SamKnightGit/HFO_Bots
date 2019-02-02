from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop
from keras.backend import eval
import tensorflow as tf
import numpy as np
import os
import time
from typing import List


class Global_QNetwork():
    def __init__(self, state_dims, learning_rate, num_teammates,
                 save_location=None, load_location=None):
        if load_location:
            self.net = load_model(load_location)
        else:
            self.net = self.create_global_network(state_dims, learning_rate, num_teammates)
        self.net._make_predict_function()

        self.save_location = save_location

    def create_global_network(self, state_dims, learning_rate, num_teammates):
        model = Sequential()
        model.add(Dense(1024, input_dim=state_dims, activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Dense(512, activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros')),
        model.add(Dense(256, activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros')),
        model.add(Dense(128, activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros')),
        model.add(Dense(2 + num_teammates, activation='softmax',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.compile(loss=mean_squared_error,
                      optimizer=RMSprop(lr=learning_rate))
        return model

    def set_save_location(self, save_location):
        self.save_location = save_location

    def save_network(self):
        if self.save_location:
            save_file = os.path.join(self.save_location, 'main_net.h5')
            open(save_file, 'w+').close()
            self.net.save(save_file)
            return
        raise AttributeError("No save locaiton (directory) specified.")


class Local_QNetwork():
    """
    Intended for use with multiple synchronous learners learning different policies.
    Based on the asynchronous q-learning algorithm proposed here:
    https://arxiv.org/abs/1602.01783
    """
    def __init__(self, architecture, weights, save_location=None, learning_rate=0.1, discount_factor=0.9,
                 epsilon=0.10, num_teammates=2, num_opponents=2):
        """

        :param state_dims: Dimensionality of state space
        :param load_location: Path where network model is loaded from
        :param save_location: Path where network model shall be saved
        """
        self.learning_rate = learning_rate

        self.main_net = self._create_model(architecture, weights)
        self.target_net = self.get_main_net_copy()

        self.discount_factor = discount_factor
        self.original_epsilon = epsilon
        self.current_epsilon = epsilon
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents

        self.save_location = save_location

    def _create_model(self, architecture, weights):
        """
        Create the network model, should only be called once during initialization

        :return:
        """
        local_network = model_from_json(architecture)
        local_network.set_weights(weights)
        local_network.compile(
            loss=mean_squared_error,
            optimizer=RMSprop(lr=self.learning_rate)
        )
        local_network._make_predict_function()
        return local_network

    def update_target_network(self, weights):
        self.target_net.set_weights(weights)

    def update_main_network(self, weights):
        self.main_net.set_weights(weights)

    def get_main_net_copy(self):
        """
        Returns a copy of the main network, used when updating target network

        :return: keras.Model
        """
        architecture = self.main_net.to_json()
        weights = self.main_net.get_weights()
        return self._create_model(architecture, weights)

    def get_target(self, experience):
        old_state, reward, state, terminal_state = experience
        target = np.array([float(reward)] * (2 + self.num_teammates))
        if not terminal_state:
            target += self.discount_factor * self.target_net.predict(state, batch_size=1)[0]
        return old_state, target.reshape((1,-1))

    def get_action(self, state):
        """
        Returns action to be taken by agent

        :param np.array state: Current state of the agent
        :return: Action with epsilon greedy policy
        :rtype: int
        """
        explore = True if np.random.random() <= self.current_epsilon else False
        if explore:
            return np.random.randint(0, 2 + self.num_teammates)

        return np.argmax(self.main_net.predict(state, batch_size=1)[0])

