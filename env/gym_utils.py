from functools import reduce
from operator import mul
from typing import Union, Tuple
import numpy as np
import gym


def get_io_size(observation_space: Union[gym.Space, gym.spaces.Tuple],
                action_space: Union[gym.Space, gym.spaces.Tuple]) -> Tuple[int, int]:
    """ Given observation and action space, return input and output sizes"""

    num_actions = get_total_space_size(action_space)
    assert num_actions > 0, 'invalid num actions'

    input_size = get_total_space_size(observation_space)
    assert input_size > 0, 'invalid input size'

    return input_size, num_actions


def is_continuous_space(space: Union[gym.Space, gym.spaces.Tuple]) -> bool:
    if isinstance(space, gym.spaces.Box):
        return True
    if isinstance(space, gym.spaces.Tuple):
        return is_tuple_of_boxes(space)
    return False


def is_tuple_of_boxes(spaces: gym.spaces.Tuple) -> bool:
    for space in spaces:
        if not isinstance(space, gym.spaces.Box):
            return False
    return True


def get_total_space_size(spaces: Union[gym.spaces.Discrete, gym.spaces.Box, gym.spaces.Tuple]) -> int:
    """ Return the total dimensionality of the gym.spaces.Box or multiple boxes
    """
    if isinstance(spaces, gym.spaces.Discrete):
        return spaces.n
    if isinstance(spaces, gym.spaces.Box):
        return get_gym_box_dimensionality(spaces)
    elif isinstance(spaces, gym.spaces.Tuple):
        return sum([get_gym_box_dimensionality(space) for space in spaces])
    raise Exception('unexpected spaces type, expected Discrete, Box or Tuple of Boxes')


def get_gym_box_dimensionality(space: gym.spaces.Box) -> int:
    """ Return the dimensionality of the gym.spaces.Box (continuous observations/actions)
    """
    if len(space.shape) == 0:
        return 1
    return reduce(mul, space.shape, 1)


def fix_observation(observation: any) -> np.ndarray:
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
    return observation


def sanitize_reward(reward: Union[float, np.ndarray]) -> float:
    # turns out that reward can be multidimensional array in some cases
    if isinstance(reward, np.ndarray):
        assert reward.size == 1
        while isinstance(reward, np.ndarray):
            reward = reward[0]
        reward = float(reward)

    return float(reward)
