import torch
from torch import nn
from typing import List, Type
from finrl_myself.torch_layers import create_mlp
from finrl_myself.weight_initialization import weight_init
from finrl_myself.optimizers import get_optimizer
from finrl_myself.get_feature_extractors import get_feature_extractor


class Actor(nn.Module):
    '''
    Sequential MLP actor
    '''
    def __init__(self,
                 net_arch: List[int] = None,
                 dropout: float = 0,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: dict = None,
                 model_params_init_kwargs: dict = None,
                 optim_aliase: str = None,
                 lr: float = None,
                 ):
        super(Actor, self).__init__()
        print(f'------actor------')
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim  # FlattenExtractor will flatten and return flatten dim
        self.dropout = dropout
        self._setup_model()
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(model_state_dict=self.state_dict(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):
        self.mu = nn.Sequential(
            *create_mlp(
                input_dim=self.features_dim,
                output_dim=self.action_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                squash_output=True,
                dropout=self.dropout,
            )
        )

    def get_actions(self, obs: torch.Tensor, deterministic=True):
        return self(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu(self.feature_extractor(obs))
