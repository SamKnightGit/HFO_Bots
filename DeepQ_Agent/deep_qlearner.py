from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop
import numpy as np

class Deep_QLearner():
    """
    Intended for use with multiple synchronous learners learning different policies.
    Based on the asynchronous q-learning algorithm proposed here:
    https://arxiv.org/abs/1602.01783
    """
    def __init__(self, state_dims, load_location, save_location, learning_rate=0.1, discount_factor=0.9, epsilon=0.10, num_teammates=2, num_opponents=2, ):
        """

        :param state_dims: Dimensionality of state space
        :param load_location: Path where network model is loaded from
        :param save_location: Path where network model shall be saved
        """
        self.state_dims = state_dims
        self.load_location = load_location
        self.save_location = save_location
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.original_epsilon = epsilon
        self.current_epsilon = epsilon
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.main_net = self._create_model()
        self.target_net = self.update_target()

    def _create_model(self):
        """
        Create the network model, should only be called once during initialization

        :return:
        """
        model = Sequential()
        model.add(Dense(30, input_dim=len(self.state_dims), activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Dense(10, activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Dense(2 + self.num_teammates, activation='sigmoid',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.compile(loss=mean_squared_error,
                      optimizer=RMSprop(lr=self.learning_rate))
        return model

    def update_target(self):
        """
        Copies the target network from the main network, updating it from previous experience

        :return:
        """
        self.main_net.save('main_net.h5')
        return load_model('main_net.h5')



    def update_main_net(self, experience_batch):
        """
        Updates the network weight's based on reward

        :param state:
        :param action:
        :param reward:
        :return:
        """
        for old_state, action, reward, state, terminal_state in experience_batch:
            target = reward
            if not terminal_state:
                target += self.discount_factor * np.amax(self.target_net.predict(state)[0])

            self.main_net.fit(old_state, target)



    def get_action(self, state):
        """
        Returns action to be taken by agent

        :param state:
        :return:
        """
        explore = True if np.random.random() <= self.current_eps else False
        if explore:
            return np.random.randint(0, 2 + self.num_teammates)
        return np.argmax(self.main_net.predict(state)[0])
