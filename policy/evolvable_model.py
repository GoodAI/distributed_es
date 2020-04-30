from abc import abstractmethod
from typing import Dict, List, Any, Union

import numpy as np
import torch
import torch.nn

from utils.available_device import my_device


class EvolvableModel:
    """Any model that can be optimized by some black-box optimization method"""

    @property
    def num_params(self) -> int:
        return self.serialize_to_genome().size

    def serialize_to_genome(self) -> np.ndarray:
        """Collects the current parameters and returns as a genome"""
        params = self.get_optimized_params()  # read nested dict containing all params to be optimzied

        result = []
        self._recursive_serialize(params, result)  # obtain list of tensors with param values

        return torch.cat(result).to('cpu').numpy()  # to genome in numpy

    def deserialize_from_genome(self, genome: np.ndarray):
        """Set parameters of the model from the given genome"""
        genome_tensor = torch.from_numpy(genome).to(my_device())

        end_pos = self._recursive_deserialize(self.get_optimized_params(), genome_tensor, pos=0)
        assert end_pos == genome.size, 'incompatible length of genome with params'

    def _recursive_serialize(self, param_source: Union[Dict, Any], result: List[torch.Tensor]):
        """Recursively go through the nested dict of params and collect list of tensors"""

        if isinstance(param_source, Dict):
            for key, value in param_source.items():  # it might be just some float
                if not isinstance(value, Dict) and not isinstance(value, torch.Tensor):
                    try:
                        result.append(torch.tensor(value, dtype=torch.float32, device=my_device()))
                    except:
                        print(f'ERROR: serialization: could not convert non-tensor parameter to float!')
                else:
                    self._recursive_serialize(value, result)
        elif isinstance(param_source, torch.Tensor):
            result.append(param_source.data.view(-1))

    def _recursive_deserialize(self,
                               params_target: Union[Dict, Any],
                               genome_source: torch.Tensor,
                               pos: int) -> int:
        """Recursively go through the nested dict of params and copy correct part of the genome to each param"""

        if isinstance(params_target, Dict):
            for key, value in params_target.items():
                if not isinstance(value, Dict) and not isinstance(value, torch.Tensor): # expect one float
                    params_target[key] = genome_source[pos].to('cpu')
                    pos += 1
                else:
                    pos = self._recursive_deserialize(value, genome_source, pos)

        elif isinstance(params_target, torch.Tensor):
            num_values = params_target.data.numel()
            params_target.data.copy_(genome_source[pos: pos+num_values].view_as(params_target.data))
            return pos + num_values

        return pos

    @abstractmethod
    def get_optimized_params(self) -> Dict[str, Union[Dict, torch.Tensor, float]]:
        """Black box optimization supported by implementing this method.

        example implementation:
        return {
            'actor': self.actor.state_dict()
        }

        return: (nested) Dict of params (tensors or potentially floats)
        """
        raise NotImplementedError()

