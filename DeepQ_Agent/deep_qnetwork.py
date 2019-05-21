from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop
import numpy as np
import os
from typing import List


class Global_QNetwork():
    def __init__(self, state_dims=0, learning_rate=0.0, num_teammates=0,
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
        model.add(Dense(2 + num_teammates, activation='linear',
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
    def __init__(self, architecture, weights, epsilon=0.0, learning_rate=0.0,
                 num_teammates=2, load_location=None):
        """
        QNetwork with no learning of q-values. Used to query actions when testing.

        :param state_dims: Dimensionality of state space
        :param load_location: Path where network model is loaded from
        :param save_location: Path where network model shall be saved
        """

        self.learning_rate = learning_rate
        if load_location:
            self.main_net = load_model(load_location)
        else:
            self.main_net = self._create_model(architecture, weights)

        self.original_epsilon = epsilon
        self.current_epsilon = epsilon
        self.num_teammates = num_teammates

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

    def get_action(self, state):
        """
        Returns action to be taken by agent

        :param np.array state: Current state of the agent
        :return: Action with epsilon greedy policy
        :rtype: (int, List[int])
        """
        explore = True if np.random.random() <= self.current_epsilon else False
        if explore:
            return np.random.randint(0, 2 + self.num_teammates), []
        qvalue_array = self.main_net.predict(state, batch_size=1)[0]
        return np.argmax(qvalue_array), qvalue_array


class Learning_QNetwork(Local_QNetwork):
    """
    Intended for use with multiple synchronous learners learning different policies.
    Based on the asynchronous q-learning algorithm proposed here:
    https://arxiv.org/abs/1602.01783
    """
    def __init__(self, architecture, weights, save_location=None, learning_rate=0.1,
                 discount_factor=0.95, epsilon=0.10, num_teammates=2):
        """

        :param state_dims: Dimensionality of state space
        :param load_location: Path where network model is loaded from
        :param save_location: Path where network model shall be saved
        """
        super().__init__(architecture, weights, epsilon, learning_rate, num_teammates)
        self.target_net = self.get_main_net_copy()

        self.discount_factor = discount_factor
        self.save_location = save_location

    def get_main_net_copy(self):
        """
        Returns a copy of the main network, used when updating target network

        :return: keras.Model
        """
        architecture = self.main_net.to_json()
        weights = self.main_net.get_weights()
        return self._create_model(architecture, weights)

    def update_target_network(self, weights):
        self.target_net.set_weights(weights)

    def update_main_network(self, weights):
        self.main_net.set_weights(weights)

    def get_target(self, experience):
        old_state, action, reward, state, terminal_state = experience
        target = np.array([0.0] * (2 + self.num_teammates))
        target[action] = reward
        if not terminal_state:
            target[action] += self.discount_factor * np.max(self.target_net.predict(state, batch_size=1)[0])
        return target.reshape((1,-1))


class Learning_DoubleQNetwork(Learning_QNetwork):

    def get_target(self, experience):
        old_state, action, reward, state, terminal_state = experience
        greedy_policy_action, _ = self.get_action(state)
        target = np.array([0.0] * (2 + self.num_teammates))
        target[action] = reward
        if not terminal_state:
            target[action] += self.discount_factor * self.target_net.predict(state, batch_size=1)[0][greedy_policy_action]
        return target.reshape((1,-1))