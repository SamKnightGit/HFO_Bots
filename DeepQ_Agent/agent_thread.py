import threading
import time
from deep_qlearner import Deep_QLearner
from deep_qplayer import Deep_QPlayer


class Learner_Thread(threading.Thread):
    def __init__(self, agent, index, finished, output_file):
        """
        :type agent: Deep_QLearner
        """
        threading.Thread.__init__(self)
        self.agent = agent
        self.index = index
        self.finished = finished #type: list
        self.output_file = output_file

    def run(self):
        self.agent.connect()
        time.sleep(5)
        self.agent.run_episodes(self.output_file)
        self.finished[self.index] = 1


class Player_Thread(threading.Thread):
    def __init__(self, agent, output_file):
        """
        :type agent: Deep_QPlayer
        """
        threading.Thread.__init__(self)
        self.agent = agent
        self.output_file = output_file

    def run(self):
        self.agent.connect()
        time.sleep(5)
        self.agent.run_episodes(self.output_file)