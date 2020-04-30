import gym


class PartiallyObservablePendulum(gym.ObservationWrapper):
    """
    Partially observable version of the classic pendulum task,
     as described in https://arxiv.org/pdf/1512.04455.pdf
    """
    def observation(self, observation):
        """
        Observation is:
            [cos(theta), sin(theta), theta')
            while the theta is zeroed here.
        """
        observation[2] = 0
        return observation

