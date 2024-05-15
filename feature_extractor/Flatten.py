from finrl_myself.utils import get_flattened_obs_dim
from finrl_myself.feature_extractor.Base import BaseFeaturesExtractor
import torch
import gym
from torch import nn


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    主要是状态需要展平，故类创建需要输入状态空间，通过状态空间维度来获取展平后的feature_dim。
    继承自BaseFeaturesExtractor

    nn.Flatten展开的维度默认从第2个维度到最后一个维度，第一个维度认为是batch,当添加时序长度维度的时候，要额外设置flatten的start_dim参数

    get_flattened_obs_dim： 利用observation_space的维度，获取展平后的一维特征向量的长度。
    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()
        self.observation_space = observation_space

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)

