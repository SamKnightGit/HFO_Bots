import threading
import time
from typing import Union, TextIO
from qlearner import QLearner
from qplayer import QPlayer


class Agent_Thread(threading.Thread):
    def __init__(self, agent: Union[QLearner, QPlayer], output_file: TextIO):
        threading.Thread.__init__(self)
        self.agent = agent
        self.output_file = output_file

    def run(self):
        self.agent.connect()
        time.sleep(5)
        self.agent.run_episodes(self.output_file)
