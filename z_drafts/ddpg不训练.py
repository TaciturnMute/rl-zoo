
# 添加到构造函数中
self.actor_params_before = self.actor.parameters()
self.actor_params_before_list = deepcopy([params.detach().numpy() for params in self.actor_params_before])


# 添加到replay之后
self.actor_params_after = self.actor.parameters()
self.actor_params_after_list = [params.detach().numpy() for params in self.actor.parameters()]
number = len([num for j in range(len(self.actor_params_after_list))
              for num in np.flatnonzero(self.actor_params_after_list[j] - self.actor_params_before_list[j])
              ]
             )
self.actor_params_before_list = deepcopy(self.actor_params_after_list)
print(f'non zero elements number:', number)