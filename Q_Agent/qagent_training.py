from hfo import *
import argparse
import os
from qlearner import QLearner
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


def reward_printer(state, action, reward):
    if action == 0:
        print("Dribble Action with reward {0} on state {1}".format(
            reward, state
        ))
    elif action == 1:
        print("Shoot Action with reward {0} on state {1}".format(
            reward, state
        ))
    else:
        print("Pass Action with reward {0} on state {1}".format(
            reward, state
        ))


def feature_printer(features, numTeammates, numOpponents):
    print("Player X Position: {0:.5f}".format(features[0]))
    print("Player Y Position: {0:.5f}".format(features[1]))
    print("Player Orientation: {0:.5f}".format(features[2]))
    print("Ball X Position: {0:.5f}".format(features[3]))
    print("Ball Y Position: {0:.5f}".format(features[4]))
    print("Able to kick: {0}".format(features[5]))
    print("Goal Center Proximity {0:.5f}".format(features[6]))
    print("Goal Center Angle: {0:.5f}".format(features[7]))
    print("Goal Opening Angle: {0:.5f}".format(features[8]))
    print("Proximity to Opponent: {0:.5f}".format(features[9]))
    for teammate in range(numTeammates):
        print("Teammate {0} Goal Opening Angle: {1:.5f}".format(
            teammate, features[10+6*teammate]))
        print("Teammate {0} Opponent Proximity: {1:.5f}".format(
            teammate, features[11 + 6 * teammate]))
        print("Teammate {0} Pass Opening Angle: {1:.5f}".format(
            teammate, features[12 + 6 * teammate]))
        print("Teammate {0} X Position: {1:.5f}".format(
            teammate, features[13 + 6 * teammate]))
        print("Teammate {0} Y Position: {1:.5f}".format(
            teammate, features[14 + 6 * teammate]))
        print("Teammate {0} Shirt Number: {1:.5f}".format(
            teammate, features[15 + 6 * teammate]))
    for opponent in range(numOpponents):
        print("Opponent {0} X Position: {1:.5f}".format(
            opponent, features[10 + 6 * numTeammates + 3 * opponent]))
        print("Opponent {0} Y Position: {1:.5f}".format(
            opponent, features[11 + 6 * numTeammates + 3 * opponent]))
        print("Opponent {0} Shirt Number: {1:.5f}".format(
            opponent, features[12 + 6 * numTeammates + 3 * opponent]))
    print("\n\n")




if __name__ == '__main__':
    Q_TABLE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/qtables/q_learner'

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--numTeammates', type=int, default=1)
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numEpisodes', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.10)
    parser.add_argument('--learningRate', type=float, default=0.10)
    parser.add_argument('--playerIndex', type=int, default=1)
    parser.add_argument('--inQTableDir', type=str, default=None)
    parser.add_argument('--outQTableDir', type=str, default=Q_TABLE_DIR)
    args = parser.parse_args()


    """ States explained as follows:
    4 states for location in quartile                       -- 4
    1 boolean state if quartile is near goal                -- 2
    Goal scoring angle, SMALL or LARGE                      -- 2
    Opponent proximity, CLOSE or FAR                        -- 2
    For each Teammate:
        1 boolean state if closer to goal than player       -- 2
        Proximity to opponent, CLOSE or FAR or INVALID      -- 3
        Pass opening angle, SMALL or LARGE or INVALID       -- 3
        Goal scoring angle, SMALL or LARGE or INVALID       -- 3
         

    OUT proximity refers to outside of the quartile of the player
    """
    NUM_STATES = 32 * (54 ** args.numTeammates)

    # Shoot, Dribble or Pass to one of N teammates or
    NUM_ACTIONS = 2 + args.numTeammates

    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=args.port)

    if args.inQTableDir:
        q_learner = QLearner(NUM_STATES, NUM_ACTIONS,
                             epsilon=args.epsilon,
                             learning_rate=args.learningRate,
                             q_table_in=args.inQTableDir + str(args.playerIndex) + '.npy',
                             q_table_out=args.outQTableDir + str(args.playerIndex) + '.npy')
    else:
        q_learner = QLearner(NUM_STATES, NUM_ACTIONS,
                             epsilon=args.epsilon,
                             learning_rate=args.learningRate,
                             q_table_in=args.outQTableDir + str(args.playerIndex) + '.npy',
                             q_table_out=args.outQTableDir + str(args.playerIndex) + '.npy')

    for episode in range(0, args.numEpisodes):
        status = IN_GAME
        action = None
        state = None
        history = []
        while status == IN_GAME:
            features = hfo.getState()
            # Print off features in a readable manner
            # feature_printer(features, args.numTeammates, args.numOpponents)

            if int(features[5]) != 1:
                history.append((features[0], features[1]))
                if len(history) > 5:
                    history.pop(0)

                # ensures agent does not get stuck for prolonged periods
                if len(history) == 5:
                    if history[0][0] == history[4][0] and history[0][1] == history[4][1]:
                        hfo.act(REORIENT)
                        history = []
                        continue

                hfo.act(MOVE)
            else:
                state, valid_teammates = state_representer.get_representation(features, args.numTeammates)
                print("Valid Teammates: ", valid_teammates)
                if 0 in valid_teammates:
                    q_learner.set_invalid(state, valid_teammates)

                if action is not None:
                    reward = get_reward(status)
                    reward_printer(state, action, reward)
                    q_learner.update(state, action, reward)

                action = q_learner.get_action(state, valid_teammates)

                if action == 0:
                    print("Action Taken: DRIBBLE \n")
                    hfo.act(DRIBBLE)
                elif action == 1:
                    print("Action Taken: SHOOT \n")
                    hfo.act(SHOOT)
                elif args.numTeammates > 0:
                    print("Action Taken: PASS -> {0} \n".format(action-2))
                    hfo.act(PASS, features[15 + 6 * (action-2)])
            status = hfo.step()

        if action is not None and state is not None:
            reward = get_reward(status)
            reward_printer(state, action, reward)
            q_learner.update(state, action, reward)
            q_learner.clear()
            q_learner.save()

        if status == SERVER_DOWN:
            hfo.act(QUIT)
            q_learner.save()
            break
            
    q_learner.save()