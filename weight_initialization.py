import torch
from torch import nn
import numpy as np


def weight_init(model_state_dict = None,
                aliase: str = 'xavier_uniform',
                activation: str = 'relu',
                a = 0,
                seed = 1,
                flag: bool = False):
    '''
    :param model_state_dict: 模型参数字典
    :param aliase: 初始化方式名称
    :param activation: 使用的激活函数
    :param a: he初始化方式需要使用的参数，default 1
    :param seed: 用来保证复现性的随机种子
    :param flag: True表示使用该函数来进行参数初始化，False表示使用torch默认的初始化方式

    :return: 参数初始化后的模型
    '''
    if not flag:
        print('random init!')
        return model_state_dict

    np.random.seed(seed)
    seeds = np.random.randint(0, 1000, 100)
    # define init methods aliases
    weight_init_aliases = {
        'xavier_uniform': nn.init.xavier_uniform_,  # no param
        'xavier_normal': nn.init.xavier_normal_,  # no param
        'he': nn.init.kaiming_normal_,  # two param
    }
    init_method = weight_init_aliases[aliase]

    # initialize each layer
    seed_order = 0
    if aliase == 'xavier_normal':
        print('xavier_normal!')
        for layer in model_state_dict:
            seed_ = seeds[seed_order]
            seed_order += 1
            torch.manual_seed(seed_)
            if layer.split('.')[-1] == 'weight':
                init_method(model_state_dict[layer])
    if aliase == 'xavier_uniform':
        print('xavier_uniform!')
        for layer in model_state_dict:
            seed_ = seeds[seed_order]
            seed_order += 1
            torch.manual_seed(seed)
            if layer.split('.')[-1] == 'weight':
                init_method(model_state_dict[layer])
    if aliase == 'he':
        print('he!')
        for layer in model_state_dict:
            seed_ = seeds[seed_order]
            seed_order += 1
            torch.manual_seed(seed)
            if layer.split('.')[-1] == 'weight':
                init_method(model_state_dict[layer], a=a, nonlinearity=activation)

    return model_state_dict

# def weight_init(model=None,
#                 aliase: str = 'xavier_uniform',
#                 activation: str = 'relu',
#                 a=0,
#                 seed=1,
#                 flag: bool = False):
#     '''
#     :param model: 需要被初始化参数的模型部分
#     :param aliase: 初始化方式名称
#     :param activation: 使用的激活函数
#     :param a: he初始化方式需要使用的参数，default 1
#     :param seed: 用来保证复现性的随机种子
#     :param flag: True表示使用该函数来进行参数初始化，False表示使用torch默认的初始化方式
#
#     :return: 参数初始化后的模型
#     '''
#     if not flag:
#         return model
#
#     np.random.seed(seed)
#     seeds = np.random.randint(0, 1000, 100)
#     # define init methods aliases
#     weight_init_aliases = {
#         'xavier_uniform': nn.init.xavier_uniform_,  # no param
#         'xavier_normal': nn.init.xavier_normal_,  # no param
#         'he': nn.init.kaiming_normal_,  # two param
#     }
#     init_method = weight_init_aliases[aliase]
#
#     # initialize each layer
#     seed_order = 0
#     if aliase == 'xavier_normal':
#         for layer in model:
#             seed_ = seeds[seed_order]
#             seed_order += 1
#             torch.manual_seed(seed_)
#             if type(layer) == nn.Linear:
#                 init_method(layer.weight)
#     if aliase == 'xavier_uniform':
#         for layer in model:
#             seed_ = seeds[seed_order]
#             seed_order += 1
#             torch.manual_seed(seed)
#             if type(layer) == nn.Linear:
#                 init_method(layer.weight)
#     if aliase == 'he':
#         for layer in model:
#             seed_ = seeds[seed_order]
#             seed_order += 1
#             torch.manual_seed(seed)
#             if type(layer) == nn.Linear:
#                 init_method(layer.weight, a=a, nonlinearity=activation)
#
#     return model