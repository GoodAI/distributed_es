import gym


class PartiallyObservableCartPole(gym.ObservationWrapper):
    """
    Partially observable version of the classic cartpole task,
     as described in https://arxiv.org/pdf/1512.04455.pdf

     cartpole: https://github.com/openai/gym/wiki/CartPole-v0
    """
    def observation(self, observation):
        """
        Observation is: [cart pos, cart vel, pole angle, pole vel at tip]
        """
        observation[1] = 0
        observation[3] = 0
        return observation


class PartiallyObservableCartPoleEasy(gym.ObservationWrapper):
    """
    Partially observable version of the classic cartpole task, only the cart velocity is hidden.
    """
    def observation(self, observation):
        """
        Observation is: [cart pos, cart vel, pole angle, pole vel at tip]
        """
        observation[3] = 0
        return observation
