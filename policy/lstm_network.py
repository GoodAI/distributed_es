from argparse import Namespace
from typing import Tuple

import torch
from torch import nn

from policy.activations import get_activation
from utils.available_device import my_device


class LSTMNetwork(nn.Module):

    hidden_size: int
    num_layers: int

    lstm: nn.LSTM
    output_layer: nn.Linear
    output_activation: nn.Module

    hidden: Tuple[torch.Tensor, torch.Tensor]

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: Namespace):
        super(LSTMNetwork, self).__init__()

        assert len(set(config.hidden_sizes)) == 1, 'hidden sizes need to be the same'

        self.hidden_size = config.hidden_sizes[0]
        self.num_layers = len(config.hidden_sizes)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers)
                            # batch_first=True)

        self.output_layer = nn.Linear(self.hidden_size, output_size)
        self.output_activation = get_activation(config.output_activation, output_size)

        self.reset()

    def forward(self, x):
        x, self.hidden = self.lstm(x.view(1, 1, -1), self.hidden)

        output = self.output_layer(x)
        output = self.output_activation(output)
        return output

    def reset(self):
        hidden_shape = (self.num_layers, 1, self.hidden_size)  # [num_layers * num_directions, batch_size, hidden_size]

        self.hidden = (
            torch.zeros(hidden_shape, device=my_device()),
            torch.zeros(hidden_shape, device=my_device())
        )

    def get_optimized_params(self):
        return self.state_dict()
