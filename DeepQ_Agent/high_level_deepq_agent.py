from hfo import *
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--numTeammates', type=int, default=1)
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numEpisodes', type=int, default=1)
    args = parser.parse_args()