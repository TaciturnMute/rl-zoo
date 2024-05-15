import torch
import copy
from torch import nn
from typing import List, Dict, Type
from finrl_myself.feature_extractor.OFENet import OFENet
from finrl_myself.weight_initialization import weight_init
from finrl_myself.optimizers import get_optimizer
from finrl_myself.torch_layers import create_mlp
from finrl_myself.get_feature_extractors import get_feature_extractor


class Actor(nn.Module):
    '''
    Sequential MLP actor, use OFENet as SRL model.
    '''
    def __init__(self,
                 net_arch: List[int] = None,
                 srl_net_arch: Dict[str, List[int]] = None,
                 dropout: float = 0,
                 state_dim: int = None,
                 action_dim: int = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: dict = None,
                 model_params_init_kwargs: dict = None,
                 optim_aliase: str = None,
                 lr: float = None,
                 activation_fn: Type[nn.Module] = None,
                 ):
        super(Actor, self).__init__()
        print(f'------actor------')
        self.net_arch = net_arch
        self.srl_net_arch = srl_net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_extractor_aliase = feature_extractor_aliase
        self.feature_extractor_kwargs = feature_extractor_kwargs
        self.dropout = dropout
        self.activation_fn = activation_fn
        self._setup_model()
        self.optim = get_optimizer(optim_aliase, self.mu.parameters(), lr)
        weight_init(model_state_dict=self.state_dict(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):
        # flatten if needed
        self.feature_extractor1 = get_feature_extractor(self.feature_extractor_aliase)
        self.feature_extractor1 = self.feature_extractor1(**self.feature_extractor_kwargs)
        self.features_dim1 = self.feature_extractor1.features_dim   # return flatten dim
        self.srl_model = OFENet(net_arch=self.srl_net_arch,
                                observation_dim=self.features_dim1,
                                action_dim=self.action_dim,
                                if_bn=False,
                                activate_fn=self.activation_fn)
        self.features_dim2 = self.srl_model.latent_obs_dim  # z_o
        self.mu = nn.Sequential(
            *create_mlp(
                input_dim=self.features_dim2,
                output_dim=self.action_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                squash_output=True,
                dropout=self.dropout,
            ))

    def get_actions(self, obs: torch.Tensor, deterministic=True) -> torch.Tensor:
        return self(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_obs = self.srl_model.phi1(self.feature_extractor1(obs))
        latent_obs = latent_obs.detach()
        return self.mu(latent_obs)
