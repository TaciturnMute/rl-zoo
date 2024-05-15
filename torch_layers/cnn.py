import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable


def create_cnn(
        input_channel: int,
        num_of_filters: List[int],
        kernels_size: List[int],
        strides: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
):
    # without pooling, BN..., to be continued...

    def cnn_block(num_in_filters: int, num_out_filters: int, kernel_size: tuple, stride: tuple) -> List[nn.Module]:
        # add a conv layer and an activation function

        layers = [nn.Conv2d(in_channels=num_in_filters,
                            out_channels=num_out_filters,
                            kernel_size=kernel_size,
                            stride=stride),
                  activation_fn()]
        return layers

    num_of_filters.insert(0, input_channel)   # note: num_of_filters will change in-place
    cnn_modules = []
    for i in range(len(num_of_filters) - 1):
        cnn_modules.extend(cnn_block(num_of_filters[i], num_of_filters[i + 1], kernels_size[i], strides[i]))
    return cnn_modules

# cnn = create_cnn(input_channel=4, num_of_filters=[32,64,64], kernels_size=[8,4,3], strides=[4,2,1])
# print(cnn)