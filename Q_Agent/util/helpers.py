from hfo import GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME


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
            teammate, features[10 + 6 * teammate]))
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


