from hfo import *
import argparse
from .q_learner import QLearner



# Taken from: high_level_sarsa_agent.py in HFO repo
def get_reward(s):
    reward = 0

    if s == GOAL:
        reward = 1

    elif s == CAPTURED_BY_DEFENSE:
        reward = -1

    elif s == OUT_OF_BOUNDS:
        reward = -1

    elif s == OUT_OF_TIME:
        reward = -1

    return reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--numTeammates', type=int, default=1)
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numEpisodes', type=int, default=1)
    args = parser.parse_args()

    """ States explained as follows:
    4 states for location in quartile                       -- 4
    1 boolean state if quartile is near goal                -- 2
    Goal scoring angle, SMALL or MED or LARGE               -- 3
    Goal scoring proximity, CLOSE or FAR or OUT             -- 3
    For each Teammate:
        1 boolean state if closer to goal than player       -- 2
        Proximity to opponent, CLOSE or FAR                 -- 2
        Pass opening angle, SMALL or LARGE or INVALID       -- 3
        Goal scoring angle, SMALL or LARGE or INVALID       -- 3
    For each Opponent:
        Proximity to said opponent, CLOSE or FAR or OUT     -- 3
         

    OUT proximity refers to outside of the quartile of the player
    """
    NUM_STATES = 12 * 10 * args.numTeammates * 3 * args.numOpponents

    # Shoot, Pass to one of N teammates or Dribble
    NUM_ACTIONS = 2 + args.numTeammates

    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=args.port)

    q_learner = QLearner(NUM_STATES, NUM_ACTIONS, q_table_out='/logs/first_table.npy')

    for episode in range(0, args.numEpisodes):
        status = IN_GAME
        while status == IN_GAME:
            features = hfo.getState()
            if int(features[5] != 1):
                hfo.act(MOVE)
            else:
                pass
            hfo.act(DRIBBLE)
            status = hfo.step()
        if status == SERVER_DOWN:
            hfo.act(QUIT)
            exit()
