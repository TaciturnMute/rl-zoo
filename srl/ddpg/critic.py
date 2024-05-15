import torch
from torch import nn
from typing import List, Dict, Type
from finrl_myself.optimizers import get_optimizer
from finrl_myself.feature_extractor.OFENet import OFENet
from finrl_myself.weight_initialization import weight_init
from finrl_myself.torch_layers import create_mlp,DualingFore_StateAction
from finrl_myself.get_feature_extractors import get_feature_extractor


class Critic(nn.Module):
    '''
    Sequential MLP, use OFENet as SRL model.
    '''
    def __init__(
            self,
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
        super(Critic, self).__init__()
        print(f'-----critic------')
        self.net_arch = net_arch
        self.srl_net_arch = srl_net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_extractor_aliase = feature_extractor_aliase
        self.feature_extractor_kwargs = feature_extractor_kwargs
        self.dropout = dropout
        self.optim_aliase = optim_aliase
        self.activation_fn = activation_fn
        self._setup_model()
        self.optim = get_optimizer(optim_aliase, self.qf.parameters(), lr)
        weight_init(self.parameters(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):
        self.feature_extractor1 = get_feature_extractor(self.feature_extractor_aliase)
        self.feature_extractor1 = self.feature_extractor1(**self.feature_extractor_kwargs)
        self.features_dim1 = self.feature_extractor1.features_dim
        self.srl_model = OFENet(net_arch=self.srl_net_arch,
                                observation_dim=self.features_dim1,
                                action_dim=self.action_dim,
                                if_bn=False,
                                activate_fn=self.activation_fn)
        self.features_dim2 = self.srl_model.latent_obs_action_dim
        self.qf = nn.Sequential(
            *create_mlp(
                input_dim=self.features_dim2,
                output_dim=1,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                dropout=self.dropout,
            )

        )

    # def f_pred(self, obs, actions):
    #     # get predictive_transition_prior, update OFENet.
    #     return self.srl_model.f_pred(self.feature_extractor1(obs), actions)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # get action value estimation
        latent_obs_action = self.srl_model.phi2(self.feature_extractor1(obs), actions)  # get z_oa
        latent_obs_action = latent_obs_action.detach()
        # assert len(states.shape) == len(actions.shape)
        return self.qf(latent_obs_action)
