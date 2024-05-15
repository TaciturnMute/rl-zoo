import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable
from finrl_myself.torch_layers import create_mlp
from finrl_myself.distributions import SquashedDiagGaussianDistribution
from finrl_myself.weight_initialization import weight_init
from finrl_myself.optimizers import get_optimizer
from finrl_myself.get_feature_extractors import get_feature_extractor

# CAP the standard deviation of the actor. exp(2) is 7.38, exp(-20) is 2e-9.
# std will be in [2e-9,7.38] which is practical.
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            activation_fn: Type[nn.Module] = None,
            feature_extractor_aliase: str = None,
            feature_extractor_kwargs: Dict = None,
            net_arch: List = [256,256],
            optim_aliase: str = None,
            lr: float = 3e-4,
            model_params_init_kwargs = None,
    ):
        super(Actor, self).__init__()
        print(f'------actor------')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.actions_dist = SquashedDiagGaussianDistribution(action_dim)
        self.model_params_init_kwargs = model_params_init_kwargs
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim  # FlattenExtractor will flatten and return flatten dim
        self._setup_model()
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(model_state_dict = self.state_dict(),**self.model_params_init_kwargs)  #model weights init
        print(f'-----------------')

    def _setup_model(self):
        # model
        self.latent_pi = nn.Sequential(*create_mlp(input_dim=self.features_dim,
                                                   output_dim=-1,
                                                   net_arch=self.net_arch,
                                                   activation_fn=self.activation_fn))
        _last_layer_dim = self.net_arch[-1]
        self.mu = nn.Linear(_last_layer_dim, self.action_dim)  # 单层
        self.log_std = nn.Linear(_last_layer_dim, self.action_dim)  # 单层

    def _get_actions_dist_params(self,
                                obs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #get mean and std params of the actions distribution
        latent_pi = self.latent_pi(self.feature_extractor(obs))    # common neural networks outputs
        mean_actions = self.mu(latent_pi)  #neural networks outputs
        log_std = self.log_std(latent_pi)  #neural networks outputs
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  #otherwise, std will be extreme, like 0 or +inf
        return (mean_actions, log_std)

    def forward(self, obs: torch.Tensor):
        # only get actions
        mean_actions, log_std = self._get_actions_dist_params(obs)
        actions = self.actions_dist.get_actions_from_params(mean_actions, log_std)  #reparameterization trick
        return actions  # range [-1,1]

    def get_actions(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            mean_actions, log_std = self._get_actions_dist_params(obs)
            actions = torch.tanh(mean_actions)
        else:
            actions = self(obs)
        return actions

    def actions_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # select action and calculate log(prob(action))
        mean_actions, log_std = self._get_actions_dist_params(obs)
        actions, log_probs = self.actions_dist.get_actions_log_prob_from_params(mean_actions,log_std)
        return (actions, log_probs)


