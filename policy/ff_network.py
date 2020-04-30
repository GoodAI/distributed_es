from argparse import Namespace
from typing import List

from torch import nn

from policy.activations import get_activation


class FFNetwork(nn.Module):

    hidden_sizes: List[int]
    output_tanh: bool

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: Namespace):
        super(FFNetwork, self).__init__()

        hidden_layers = []
        layer_activations = []

        self.layer_sizes = config.hidden_sizes + [output_size]
        activations = [config.hidden_activation] * len(config.hidden_sizes) + [config.output_activation]

        previous_size = input_size

        for layer_size, activation in zip(self.layer_sizes, activations):

            hidden_layers.append(nn.Linear(previous_size, layer_size))
            layer_activations.append(get_activation(activation, layer_size))

            previous_size = layer_size

        # register the parameters
        self.layers = nn.ModuleList(hidden_layers)
        self.activations = nn.ModuleList(layer_activations)

    def forward(self, x):
        for layer_id in range(len(self.layer_sizes)):
            x = self.layers[layer_id](x)
            x = self.activations[layer_id](x)
        return x

    def reset(self):
        pass

    def get_optimized_params(self):
        return self.state_dict()
