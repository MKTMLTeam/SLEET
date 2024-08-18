from torch.nn import functional as F
from typing import Callable
from torch import nn
from .base import Dense
from typing import Union
from typing import Sequence
import torch
from typing import Optional

__all__ = ['build_mlp', ]


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    last_bias: bool = True,
    last_zero_init: bool = False,
    atomwise_bias: bool = True,
) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation, bias=atomwise_bias)
        for i in range(n_layers - 1)
    ]
    # assign a Dense layer (without activation function) to the output layer

    if last_zero_init:
        layers.append(
            Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                bias=last_bias,
            )
        )
    else:
        layers.append(
            Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=last_bias)
        )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net



