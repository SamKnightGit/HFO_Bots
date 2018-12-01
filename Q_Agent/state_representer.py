import numpy as np


def get_representation(state_arr, num_teammates):
    """
    :param state_arr: Array of raw state returned from the HFO environment
    :param num_teammates: Used for indexing in the raw state array
    :param num_opponents: Used for indexing in the raw state array
    :return: The index of the current state in the Q Table
    :rtype: int
    """
    agent_x = state_arr[0]
    agent_y = state_arr[1]
    ball_kickable = state_arr[5]
    goal_angle = state_arr[8]
    prox_opponent = state_arr[9]
    teammates = {}
    for x in range(0, num_teammates):
        index = 10 + 6 * x
        teammates[x] = {
            'goal_angle': state_arr[index],
            'prox_opponent': state_arr[index+1],
            'pass_angle': state_arr[index+2],
            'team_x': state_arr[index+3],
            'team_y': state_arr[index+4],
            'uniform_num': state_arr[index+5]
        }

    index = 0
    previous_size = 0

    position, in_goal_region = position_finder(agent_x, agent_y)

    index += position
    previous_size += 4

    index += in_goal_region * previous_size
    previous_size += 2

    if abs(goal_angle) > 0.2:
        index += previous_size
    previous_size += 2

    if prox_opponent > 0.7:
        index += previous_size
    previous_size += 2

    valid_teammates = [0] * num_teammates
    for teammate in teammates.keys():
        valid, further_than_agent, close_to_opp, pass_angle, goal_angle = \
            get_teammate_metrics(np.array([agent_x, agent_y]), teammates[teammate])

        if valid:
            valid_teammates[teammate] = 1

        index += (previous_size * further_than_agent)
        previous_size += 2

        index += (previous_size * close_to_opp)
        previous_size += 3

        index += (previous_size * pass_angle)
        previous_size += 3

        index += (previous_size * goal_angle)
        previous_size += 3

    return index, valid_teammates

def position_finder(x_pos, y_pos):
    """
    :param float x_pos: X position of agent
    :param float y_pos: Y position of agent
    :return: Q Table index of agent position.
    Position of agent in terms of quartile block:
        1 == Top Left, 2 == Top right, 3 == Bottom Left, 4 == Bottom Right
    Multiplied by 1 if agent is not on goal side of pitch and 2 otherwise.
    :rtype: int
    """
    pos_grid = np.zeros((2,2))
    in_goal_region = 0
    y_pos = abs(y_pos)
    if x_pos > 0:
        in_goal_region = 1
        if x_pos > 0.5:
            if y_pos > 0.5:
                pos_grid[1][1] = 1.0
            else:
                pos_grid[0][1] = 1.0
        else:
            if y_pos > 0.5:
                pos_grid[1][0] = 1.0
            else:
                pos_grid[0][0] = 1.0
    else:
        if x_pos < -0.5:
            if y_pos > 0.5:
                pos_grid[1][1] = 1.0
            else:
                pos_grid[0][1] = 1.0
        else:
            if y_pos > 0.5:
                pos_grid[1][0] = 1.0
            else:
                pos_grid[0][0] = 1.0

    return np.flatnonzero(pos_grid)[0], in_goal_region


def get_teammate_metrics(agent_pos, teammate):
    """
    For supplied teammate returns the index into Q Table by the metrics:
    1) 1 if farther from goal than agent, 2 if closer
    2) 1 if close to opponent, 2 if not (close defined by in same quartile), 3 if invalid
    3) 1 if pass opening angle is small, 2 if large, 3 if invalid
    4) 1 if goal angle is small, 2 if large, 3 if invalid

    :param numpy.array agent_pos: Agent position
    :param dict teammate: Teammate information
    :return: Index into Q Table for teammate
    :rtype: (int, bool)
    """
    goal_pos = np.array([1.0, 0.0])
    teammate_pos = np.array([teammate['team_x'], teammate['team_y']])
    team_dist = np.linalg.norm(teammate_pos-goal_pos)
    agent_dist = np.linalg.norm(agent_pos-goal_pos)

    if teammate_pos[0] == -2 or teammate_pos[1] == -2:
        valid = False
    else:
        valid = True

    further_than_agent = 1
    if team_dist < agent_dist:
        further_than_agent = 0

    close_to_opp = 2
    prox_opponent = teammate['prox_opponent']
    if prox_opponent != -2:
        if prox_opponent < 0.7:
            close_to_opp = 0
        else:
            close_to_opp = 1

    # TODO: Understand what angles make sense
    pass_angle = 2
    p_angle = teammate['pass_angle']
    if p_angle != -2:
        if abs(p_angle) < 0.2:
            pass_angle = 0
        else:
            pass_angle = 1

    goal_angle = 2
    g_angle = teammate['goal_angle']
    if g_angle != -2:
        if abs(g_angle) < 0.2 :
            goal_angle = 0
        else:
            goal_angle = 1

    return valid, further_than_agent, close_to_opp, pass_angle, goal_angle









