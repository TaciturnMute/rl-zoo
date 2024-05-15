import torch
import gym
from torch import nn


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.
    从状态空间中获取特征维度
    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()