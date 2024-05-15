import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable


class DensenetBolck(nn.Module):

    def __init__(self, input_dim, output_dim, activation_fn, kernel_initializer="glorot_uniform", batchnorm=False):
        super(DensenetBolck, self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)
        self.af = activation_fn()

    def forward(self, input):
        output = self.fc(input)
        output = self.af(output)
        return torch.concat((input,output),dim=1)


# densenet_block = Densenet_Bolck(input_dim=128,output_dim=128,activation_fn=nn.ReLU)
# print(densenet_block)
# print(densenet_block(torch.randn(64,128)).shape)