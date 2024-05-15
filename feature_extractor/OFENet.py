import torch
from torch import nn
from typing import Dict, List, Type, Tuple
from finrl_myself.torch_layers import create_mlp


class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super(ConcatLayer, self).__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=self.dim)


class OFENet(nn.Module):
    '''
    MLP Dense Net
    if net_arch['phi1_net'] is None, Zo is O
    if net_arch['phi2_net'] is None, Zoa is concat(Zo,a)
    if net_arch['f_pred_net'] is None, o_t+1_pre is Zoa
    '''
    def __init__(
            self,
            net_arch: Dict[str, List[int]],
            observation_dim: int,
            action_dim: int,
            if_bn: bool,
            activate_fn: Type[nn.Module] = nn.Tanh,
    ):
        super(OFENet, self).__init__()
        self.net_arch = net_arch
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.if_bn = if_bn
        self.activate_fn = activate_fn
        self._setup_model()

    def _setup_model(self) -> None:
        # {'phi1':[400,300],'phi2':[400,300], 'f_pred':[300,200]}
        phi1_only_layers = []
        phi2_only_layers = []
        f_pred_only_layers = []
        for layer in self.net_arch.keys():
            # TODO: give layer a meaningful name
            if layer == 'phi1':
                assert isinstance(self.net_arch[layer], list), \
                    "Error: OFENet net_arch['phi1'] must contain a list of integers."
                phi1_only_layers = self.net_arch['phi1']
            if layer == 'phi2':
                assert isinstance(self.net_arch[layer], list), \
                    "Error: OFENet net_arch['phi2'] must contain a list of integers."
                phi2_only_layers = self.net_arch['phi2']
            if layer == 'f_pred':
                assert isinstance(self.net_arch[layer], list), \
                    "Error: OFENet net_arch['f_pred'] must contain a list of integers."
                f_pred_only_layers = self.net_arch['f_pred']

        phi1_net = []
        phi2_net = []

        # create phi1
        phi1_last_concat_dim = self.observation_dim
        if len(phi1_only_layers) > 0:
            for phi1_layer_dim in phi1_only_layers:
                phi1_net.append(nn.Linear(phi1_last_concat_dim, phi1_layer_dim))
                phi1_net.append(self.activate_fn())
                phi1_last_concat_dim += phi1_layer_dim
                if self.if_bn:
                    phi1_net.append(nn.BatchNorm1d(num_features=phi1_last_concat_dim))
        self.phi1_net = nn.Sequential(*phi1_net)

        # create phi2
        phi2_last_concat_dim = phi1_last_concat_dim + self.action_dim
        if len(phi2_only_layers) > 0:
            for phi2_layer_dim in phi2_only_layers:
                phi2_net.append(nn.Linear(phi2_last_concat_dim, phi2_layer_dim))
                phi2_net.append(self.activate_fn())
                phi2_last_concat_dim += phi2_layer_dim
                if self.if_bn:
                    phi2_net.append(nn.BatchNorm1d(num_features=phi2_last_concat_dim))
        self.phi2_net = nn.Sequential(*phi2_net)

        # create f_pred
        f_pred_input_dim = phi2_last_concat_dim
        f_pred_output_dim = self.observation_dim
        self.f_pred_net = nn.Sequential(
            *create_mlp(
                input_dim=f_pred_input_dim,
                output_dim=f_pred_output_dim,
                net_arch=f_pred_only_layers,
                activation_fn=self.activate_fn,
            )
        )

        self.concat = ConcatLayer(dim=1)
        self.latent_obs_dim = phi1_last_concat_dim  # used for actor
        self.latent_obs_action_dim = phi2_last_concat_dim  # used for critic

    def phi1(self, observations) -> torch.Tensor:
        if len(self.phi1_net) == 0:
            latent_obs = observations
        else:
            layer_inputs = observations
            for _, layer in self.phi1_net.named_children():
                # print(name, layer)
                # each step in phi net is: output = activation(linear_output + input)
                if isinstance(layer, nn.Linear):
                    concat_outputs = self.concat(layer(layer_inputs), layer_inputs)
                    layer_inputs = concat_outputs
                    latent_obs = concat_outputs
                else:
                    concat_outputs = layer(layer_inputs)
                    layer_inputs = concat_outputs
                    latent_obs = concat_outputs
        return latent_obs

    def phi2(self, observations, actions) -> torch.Tensor:
        return self.forward(observations, actions)[1]

    def f_pred(self, observations, actions) -> torch.Tensor:
        return self.forward(observations, actions)[2]

    def forward(self, observations, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_obs = self.phi1(observations)
        phi2_layer_inputs = self.concat(latent_obs, actions)
        if len(self.phi2_net) == 0:
            latent_obs_action = phi2_layer_inputs
        else:
            for _, layer in self.phi2_net.named_children():
                if isinstance(layer, nn.Linear):
                    phi2_concat_outputs = self.concat(layer(phi2_layer_inputs), phi2_layer_inputs)
                    phi2_layer_inputs = phi2_concat_outputs
                    latent_obs_action = phi2_concat_outputs
                else:
                    phi2_concat_outputs = layer(phi2_layer_inputs)
                    phi2_layer_inputs = phi2_concat_outputs
                    latent_obs_action = phi2_concat_outputs
        predictive_transition_prior = self.f_pred_net(latent_obs_action)  # o_t+1_pred
        return latent_obs, latent_obs_action, predictive_transition_prior

# ofe = OFENet({'phi1':[717]*4,'phi2':[957]*4,'f_pred':[]},957,28,False)
# ofe = OFENet({'phi1':[300,200],'phi2':[200,100]},289,28, True)
# ofe = OFENet({'phi1':[],'phi2':[]},289,28)      # only concat observations and actions
# print(ofe)
# observations = torch.randn(64, 289)
# actions = torch.randn(64, 28)
# print(ofe.phi1(observations).shape)
# print(ofe.phi2(observations, actions).shape)
# print(ofe.f_pred(observations, actions).shape)
