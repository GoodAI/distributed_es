from argparse import Namespace
from pydoc import locate
from typing import Dict, Union

import gym
import numpy as np
import torch
# from badger_utils import SacredWriter, SacredReader, Serializable
from badger_utils.sacred import Serializable, SacredWriter, SacredReader

from policy.evolvable_model import EvolvableModel
from policy.ff_network import FFNetwork
from utils.available_device import my_device
from env.gym_utils import get_io_size


class Policy(Serializable, EvolvableModel):
    actor: FFNetwork

    input_size: int
    num_actions: int

    actions_discrete: bool

    def __init__(self,
                 observation_space: Union[gym.Space, gym.spaces.Tuple],
                 action_space: Union[gym.Space, gym.spaces.Tuple],
                 config: Namespace,
                 _run=None):
        self.input_size, self.num_actions = get_io_size(observation_space, action_space)
        self.actions_discrete = isinstance(action_space, gym.spaces.Discrete)

        network_class = locate(config.network)

        self.actor = network_class(self.input_size, self.num_actions, config).to(my_device())

    def pick_action(self, observation: np.ndarray) -> np.array:
        assert observation.size == self.input_size

        obs = torch.tensor(observation, dtype=torch.float32, device=my_device()).view(-1)
        action = self.actor.forward(obs)
        action = action.detach().to('cpu').numpy().reshape(-1)

        if self.actions_discrete:
            return int(np.argmax(action))

        return action

    def reset(self):
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

    def save(self, writer: SacredWriter, epoch: int):
        writer.save_model(model=self, name=self.name, epoch=epoch)

    def load(self, reader: SacredReader, epoch: int):
        reader.load_model(model=self, name=self.name, epoch=epoch)

    def serialize(self) -> Dict[str, object]:
        result = {'actor': self.actor.state_dict()}
        return result

    def deserialize(self, data: Dict[str, object]):
        data: Dict[str, Dict[str, torch.Tensor]]
        self.actor.load_state_dict(data['actor'])
        print(f'Policy deserialized correctly')

    def get_optimized_params(self) -> Dict[str, object]:
        result = {
            'actor': self.actor.get_optimized_params()
        }
        return result
