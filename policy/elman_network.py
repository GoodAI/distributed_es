from argparse import Namespace

import torch
from torch import nn

from policy.activations import get_activation
from utils.available_device import my_device


class ElmanNetwork(nn.Module):

    hidden_size: int

    hidden_layer: nn.Linear
    output_layer: nn.Linear
    output_activation: nn.Module

    hidden: torch.Tensor

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: Namespace):
        super(ElmanNetwork, self).__init__()

        assert len(config.hidden_sizes) == 1, 'only 1 hidden size supported'

        self.hidden_size = config.hidden_sizes[0]

        self.hidden_layer = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.hidden_activation = get_activation(config.hidden_activation, self.hidden_size)

        self.output_layer = nn.Linear(self.hidden_size, output_size)
        self.output_activation = get_activation(config.output_activation, output_size)

        self.reset()

    def forward(self, x):
        total_input = torch.cat([x.view(-1), self.hidden])

        self.hidden = self.hidden_activation(self.hidden_layer(total_input))
        output = self.output_activation(self.output_layer(self.hidden))

        return output

    def reset(self):
        self.hidden = torch.zeros(self.hidden_size, device=my_device())

    def get_optimized_params(self):
        return self.state_dict()
