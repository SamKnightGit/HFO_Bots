# Hand engineered reward functions
import numpy as np
from hfo import GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME

goal_position = np.array([1.0, 0.0])
max_distance_from_goal = np.linalg.norm(
    np.array([-1.0, 1.0]) - goal_position
)


def get_sparse_reward(status):
    reward = 0  # type: int

    if status == GOAL:
        reward = 1

    elif status == CAPTURED_BY_DEFENSE:
        reward = -1

    elif status == OUT_OF_BOUNDS:
        reward = -1

    elif status == OUT_OF_TIME:
        reward = -1

    return reward


def discrete_simple_reward(old_state, action, state):
    pass


def discrete_advanced_reward(old_state, action, state):
    pass


def hl_simple_reward(old_state, action, state):
    old_state = old_state[0]
    state = state[0]
    if _in_scoring_zone(old_state):
        return -0.49 + _scoring_reward(action)
    return -0.99 + _distance_reward(state)


def _scoring_reward(action):
    if action == 1: # shot taken
        return 0.49
    return 0


def _in_scoring_zone(old_state):
    old_position = np.array([old_state[0], old_state[1]])
    goal_dist = np.linalg.norm(old_position - goal_position)
    if goal_dist <= 0.5:
        return True
    return False


def _distance_reward(state):
    new_position = np.array([state[0], state[1]])
    new_goal_dist = np.linalg.norm(new_position - goal_position)
    distance_ratio = (new_goal_dist / max_distance_from_goal) * 0.50
    return 0.50 - distance_ratio


def hl_advanced_reward(old_state, action, state):
    pass


def ll_simple_reward(old_state, action, state):
    pass


def ll_advanced_reward(old_state, action, state):
    pass
