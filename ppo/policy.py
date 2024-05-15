import torch
from torch import nn
from finrl_myself.torch_layers import *
from finrl_myself.distributions import SquashedDiagGaussianDistribution
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable
from finrl_myself.optimizers import get_optimizer
from finrl_myself.weight_initialization import weight_init
from finrl_myself.get_feature_extractors import get_feature_extractor


class DualingPostPolicy(nn.Module):

    def __init__(self,
                 net_arch: List[Union[int, Dict[str,List[int]]]] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: Dict = None,
                 log_std_init: float = 0.0,
                 optim_aliase: str = None,
                 lr: float = 0.0003,
                 model_params_init_kwargs: dict = None,
                 ):
        super(DualingPostPolicy, self).__init__()
        print(f'------policy------')
        self.net_arch = net_arch
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim  # FlattenExtractor will flatten and return flatten dim
        self.log_std_init = log_std_init
        self._setup_model()
        self.dist = SquashedDiagGaussianDistribution(actions_dim = self.action_dim)
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(model_state_dict=self.state_dict(), **model_params_init_kwargs)
        print(f'------------------')

    def _setup_model(self):
        self.mlp_extractor = DualingPost_ActorCritic(
            net_arch=self.net_arch,
            input_dim=self.features_dim,
            activation_fn=self.activation_fn,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_vf_dim, 1)  # critic
        self.action_net = nn.Linear(self.mlp_extractor.latent_pi_dim, self.action_dim) # actor
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * self.log_std_init, requires_grad = True)  # a number  

    def predict_values(self, obs: torch.Tensor):
        # critic part
        # only get value estimation, PPO collect_rollout function use
        states = self.feature_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(states)
        values = self.value_net(latent_vf)
        return values

    def get_actions(self, obs: torch.Tensor, deterministic=False):
        if deterministic:
            states = self.feature_extractor(obs)
            latent_pi = self.mlp_extractor.forward_actor(states)
            mean_actions = self.action_net(latent_pi)
            actions = torch.tanh(mean_actions)
        else:
            # actor part
            states = self.feature_extractor(obs)
            latent_pi = self.mlp_extractor.forward_actor(states)
            mean_actions = self.action_net(latent_pi)
            actions = self.dist.get_actions_from_params(mean_actions, self.log_std)
        return actions

    def get_gaussian_actions(self):
        return self.dist.gaussian_actions

    def forward(self, obs: torch.Tensor):
        # get actions, values, log_prob
        states = self.feature_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(states)
        values = self.value_net(latent_vf)
        mean_actions = self.action_net(latent_pi)
        actions, log_prob = self.dist.get_actions_log_prob_from_params(mean_actions, self.log_std)
        return actions, values, log_prob  #torch.tensor,torch.tensor,torch.tensor

    def evaluate_actions(self,
                         obs: torch.Tensor,
                         actions: torch.Tensor,
                         gaussian_actions: torch.Tensor):
        # Evaluate actions according to the current policy, given the observations.
        states = self.feature_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(states)
        values = self.value_net(latent_vf)  # estimate current state values
        mean_actions = self.action_net(latent_pi)
        # get current policy log(pi(at|st)) 1. get current policy from given params(each state==> each mean_actions==> given params) 2. get log prob for given action
        log_prob = self.dist.get_log_prob_from_actions(actions=actions,
                                                       gaussian_actions=gaussian_actions,
                                                       mean_actions=mean_actions,
                                                       log_std=self.log_std)
        log_prob = log_prob.reshape(-1, 1)
        entropy = self.dist.dist.entropy()
        entropy = entropy.reshape(-1, 1)
        return values, log_prob, entropy   # (batch_size, 1)


# CAP the standard deviation of the actor. exp(2) is 7.38, exp(-20) is 2e-9.
# std will be in [2e-9,7.38] which is practical.
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class D2RLPolicy(nn.Module):

    def __init__(self,
                 net_arch: List[Union[int, Dict[str,List[int]]]] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: Dict = None,
                 log_std_init: float = 0.0,
                 optim_aliase: str = None,
                 lr: float = None,
                 model_params_init_kwargs: dict = None,
                 ):
        super(D2RLPolicy, self).__init__()
        print(f'------policy------')
        self.net_arch = net_arch
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim  # FlattenExtractor will flatten and return flatten dim
        # self.log_std_init = log_std_init
        self._setup_model()
        self.dist = SquashedDiagGaussianDistribution(actions_dim = self.action_dim)
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(model_state_dict=self.state_dict(), **model_params_init_kwargs)
        print(f'------------------')

    def _setup_model(self):
        # self.mlp_extractor = DualingPost_ActorCritic(
        #     net_arch=self.net_arch,
        #     input_dim=self.features_dim,
        #     activation_fn=self.activation_fn,
        # )
        self.mlp_extractor = D2RL(
            input_dim=self.features_dim,
            hidden_dims=self.net_arch,
            activation_fn=self.activation_fn,
        )
        self.latent_dim = self.net_arch[-1] + self.features_dim
        self.value_net = nn.Linear(self.latent_dim, 1)  # critic
        self.action_mean_net = nn.Linear(self.latent_dim, self.action_dim) # actor
        self.action_log_std_net = nn.Linear(self.latent_dim, self.action_dim) # actor
        # self.log_std = nn.Parameter(torch.ones(self.action_dim) * self.log_std_init, requires_grad = True)  # a vector

    def predict_values(self, obs: torch.Tensor):
        # critic part
        # only get value estimation, PPO collect_rollout function use
        states = self.feature_extractor(obs)
        latent = self.mlp_extractor(states)
        values = self.value_net(latent)
        return values

    def get_actions(self, obs: torch.Tensor, deterministic=False):
        if deterministic:
            states = self.feature_extractor(obs)
            latent = self.mlp_extractor(states)
            mean_actions = self.action_mean_net(latent)
            actions = torch.tanh(mean_actions)
        else:
            # actor part
            states = self.feature_extractor(obs)
            latent = self.mlp_extractor(states)
            mean_actions = self.action_mean_net(latent)
            log_std = torch.clamp(self.action_log_std_net(latent), LOG_STD_MIN, LOG_STD_MAX)
            actions = self.dist.get_actions_from_params(mean_actions, log_std)
        return actions

    def get_gaussian_actions(self):
        return self.dist.gaussian_actions

    def forward(self, obs: torch.Tensor):
        # get actions, values, log_prob
        states = self.feature_extractor(obs)
        latent = self.mlp_extractor(states)
        values = self.value_net(latent)
        mean_actions = self.action_mean_net(latent)
        log_std = torch.clamp(self.action_log_std_net(latent), LOG_STD_MIN, LOG_STD_MAX)
        actions, log_prob = self.dist.get_actions_log_prob_from_params(mean_actions, log_std)
        return actions, values, log_prob  #torch.tensor,torch.tensor,torch.tensor

    def evaluate_actions(self,
                         obs: torch.Tensor,
                         actions: torch.Tensor,
                         gaussian_actions: torch.Tensor):
        # Evaluate actions according to the current policy, given the observations.
        states = self.feature_extractor(obs)
        latent= self.mlp_extractor(states)
        values = self.value_net(latent)  # estimate current state values
        mean_actions = self.action_mean_net(latent)
        log_std = torch.clamp(self.action_log_std_net(latent), LOG_STD_MIN, LOG_STD_MAX)
        # get current policy log(pi(at|st))
        log_prob = self.dist.get_log_prob_from_actions(actions=actions,
                                                       gaussian_actions=gaussian_actions,
                                                       mean_actions=mean_actions,
                                                       log_std=log_std)
        log_prob = log_prob.reshape(-1, 1)
        entropy = self.dist.dist.entropy()
        entropy = entropy.reshape(-1, 1)
        return values, log_prob, entropy   # (batch_size, 1)
