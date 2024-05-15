import torch
from torch import nn
import collections
from collections import OrderedDict

from finrl_myself.optimizers import get_optimizer
from finrl_myself.weight_initialization import weight_init
from finrl_myself.feature_extractor.OFENet import OFENet
from finrl_myself.get_feature_extractors import get_feature_extractor
from finrl_myself.torch_layers import create_mlp, DualingFore_StateAction
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable


class Critic(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: Dict = None,
                 net_arch:List = [400,300],
                 srl_net_arch: Dict[str, List[int]] = None,
                 optim_aliase:str = None,
                 lr = 3e-4,
                 model_params_init_kwargs = None,
                 n_critics:int = 2,
                 ):
        super(Critic,self).__init__()
        print(f'-----critic------')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_arch = net_arch   #sequential network architecture, very simple
        self.srl_net_arch = srl_net_arch
        self.n_critics = n_critics
        self.activation_fn = activation_fn
        self.optim_aliase = optim_aliase
        self.feature_extractor_aliase = feature_extractor_aliase
        self.feature_extractor_kwargs = feature_extractor_kwargs
        self._setup_model()
        self.optim1 = get_optimizer(optim_aliase, self.qf1.parameters(), lr)
        self.optim2 = get_optimizer(optim_aliase, self.qf2.parameters(), lr)
        weight_init(model_state_dict=self.state_dict(), **model_params_init_kwargs)  # model weights init
        print(f'-----------------')

    def _setup_model(self):
        # critic model: one srl model, two qf
        self.feature_extractor1 = get_feature_extractor(self.feature_extractor_aliase)
        self.feature_extractor1 = self.feature_extractor1(**self.feature_extractor_kwargs)
        self.features_dim1 = self.feature_extractor1.features_dim
        self.srl_model = OFENet(net_arch=self.srl_net_arch,
                                observation_dim=self.features_dim1,
                                action_dim=self.action_dim,
                                if_bn=False,
                                activate_fn=self.activation_fn)
        self.features_dim2 = self.srl_model.latent_obs_action_dim
        for idx in range(1, self.n_critics+1):
            qf = nn.Sequential(*create_mlp(input_dim=self.features_dim2,
                                          output_dim=1,
                                          net_arch=self.net_arch,
                                          activation_fn=self.activation_fn))

            self.add_module(f'qf{idx}',qf)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        latent_obs_action = self.srl_model.phi2(self.feature_extractor1(obs), actions)  # get z_oa
        latent_obs_action = latent_obs_action.detach()
        return (self.qf1(latent_obs_action), self.qf2(latent_obs_action))

    def get_q_values(self, obs: torch.Tensor, actions: torch.Tensor):
        # _inputs = torch.cat((obs, actions),dim=1)
        return self(obs, actions)