# 添加到replay中

# optim overall
# self.critic.optim.zero_grad()
# # assert q_value_pres1.shape==q_value_pres2.shape==td_targets.shape==(self.batch_size ,1)
# critic_loss = nn.functional.mse_loss(q_value_pres1, td_targets) + nn.functional.mse_loss(q_value_pres2, td_targets)
# critic_loss.backward()
# self.critic.optim.step()



# class Critic_DualingFore(nn.Module):
#     '''
#     Dualing MLP
#     '''
#     def __init__(self,
#                  net_arch: List[Union[int, Dict[str, List[int]]]] = None,
#                  state_dim: int = None,
#                  action_dim: int = None,
#                  activation_fn: Type[nn.Module] = None,
#                  feature_extractor_aliase: str = None,
#                  feature_extractor_kwargs: dict = None,
#                  model_params_init_kwargs: dict = None,
#                  optim_aliase: str = None,
#                  lr: float = None,
#                  n_critics: int = 2
#                  ):
#
#         super(Critic_DualingFore, self).__init__()
#         print(f'-----critic------')
#         self.net_arch = net_arch
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.n_critics = n_critics
#         self.activation_fn = activation_fn
#         self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
#         self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
#         self.features_dim = self.feature_extractor.features_dim
#         self._setup_model()
#         self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
#         weight_init(self.parameters(), **model_params_init_kwargs)
#         print(f'-----------------')
#
#     def _setup_model(self):
#         # two q_network for clipped double q-learning algorithm
#         for idx in range(1, self.n_critics+1):
#             dualing_mlp = DualingFore_StateAction(
#                 state_dim=self.features_dim,
#                 action_dim=self.action_dim,
#                 net_arch=self.net_arch,
#                 activation_fn=self.activation_fn,
#             )
#             q_net = nn.Linear(dualing_mlp.last_layer_dim_shared_net, 1)
#             self.add_module(f'dualing_mlp{idx}',dualing_mlp)
#             self.add_module(f'qf{idx}',q_net)
#
#     def get_qf1_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
#         states = self.feature_extractor(obs)
#         latent_feature = self.dualing_mlp1(states, actions)
#         return self.qf1(latent_feature)
#
#     def forward(self,
#                 obs: torch.Tensor,
#                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         states = self.feature_extractor(obs)
#         latent_feature1 = self.dualing_mlp1(states, actions)
#         latent_feature2 = self.dualing_mlp2(states, actions)
#         q1 = self.qf1(latent_feature1)
#         q2 = self.qf2(latent_feature2)
#         return (q1, q2)