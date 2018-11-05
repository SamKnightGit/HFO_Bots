from hfo import *
import argparse
import os
from q_learner import QLearner
import state_representer


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
    parser.add_argument('--playerIndex', type=int, default=1)
    args = parser.parse_args()


    """ States explained as follows:
    4 states for location in quartile                       -- 4
    1 boolean state if quartile is near goal                -- 2
    Goal scoring angle, SMALL or LARGE                      -- 2
    Opponent proximity, CLOSE or FAR                        -- 2
    For each Teammate:
        1 boolean state if closer to goal than player       -- 2
        Proximity to opponent, CLOSE or FAR                 -- 2
        Pass opening angle, SMALL or LARGE or INVALID       -- 3
        Goal scoring angle, SMALL or LARGE or INVALID       -- 3
         

    OUT proximity refers to outside of the quartile of the player
    """
    NUM_STATES = 32 * (36 ** args.numTeammates)

    # Shoot, Pass to one of N teammates or Dribble
    NUM_ACTIONS = 2 + args.numTeammates

    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=args.port)

    logs_dir = os.path.dirname(os.path.realpath(__file__))
    q_learner = QLearner(NUM_STATES, NUM_ACTIONS,
                         q_table_in = logs_dir + '/logs/q_learner' + str(args.playerIndex) + '.npy',
                         q_table_out=logs_dir + '/logs/q_learner' + str(args.playerIndex) + '.npy')

    for episode in range(0, args.numEpisodes):
        status = IN_GAME
        action = None
        state = None
        timestep = 0
        while status == IN_GAME:
            timestep += 1
            features = hfo.getState()
            if int(features[5] != 1):
                hfo.act(MOVE)
            else:
                state = state_representer.get_representation(features, args.numTeammates)
                if action:
                    reward = get_reward(status)
                    q_learner.update(state, action, reward)
                action = q_learner.get_action(state)

                if action == 0:
                    hfo.act(DRIBBLE)
                elif action == 1:
                    hfo.act(SHOOT)
                elif args.numTeammates > 0:
                    hfo.act(PASS, 15 + 6 * (action-2))
            status = hfo.step()

        if action and state:
            reward = get_reward(status)
            q_learner.update(state, action, reward)
            q_learner.clear()

        if status == SERVER_DOWN:
            hfo.act(QUIT)
            q_learner.save()
            break
            
    q_learner.save()