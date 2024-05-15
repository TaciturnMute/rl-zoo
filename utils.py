from itertools import zip_longest
from typing import Iterable,Union
import numpy as np
import torch
import copy
import pandas as pd
from copy import deepcopy
from gym import spaces
import json
import matplotlib.pyplot as plt


"""
zip_strict
data_split
sum_independent_dims
CompleteShape
polyak_update
load_checkpoint
get_flattened_obs_dim
save_params
set_mode
training_plot
"""

def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo

def data_split(df: pd.Series, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(pd.to_datetime(df[target_date_col]) >= pd.to_datetime(start)) & (pd.to_datetime(df[target_date_col]) < pd.to_datetime(end))]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


def CompleteShape(data: np.ndarray):

    batch_size = data.shape[0]  # first dim of data is batch_size by default
    if len(data.shape) == 2:    # each data in the batch is a vector or a scalar
        # if data.shape[1] > 1:     # vector
        #     dim1 = data.shape[1]
        #     data = data.reshape(batch_size, 1, dim1)
        #     return data
        # else:
        #     return data           # scalar and has complete shape
        return data
    elif len(data.shape) == 1:   # each data in this batch is scalar but has no complete shape
        return data.reshape(batch_size, 1)
    else: # each data in the batch is at least two or higher dimensionality, do not need to process
        return data


def polyak_update(params:Iterable[torch.Tensor],target_params:Iterable[torch.Tensor],tau)->None:
    #execute soft update
    with torch.no_grad():
        for param,target_param in zip_strict(params,target_params):
            target_param.data.mul_(1 - tau)
            torch.add(input = target_param.data, other = param.data, alpha = tau, out = target_param.data)

def load_checkpoint(checkpoint:dict,agent):
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor.optim.load_state_dict(checkpoint['actor_optim_state_dict'])
    agent.critic.optim.load_state_dict(checkpoint['critic_optim_state_dict'])
    agent.critic_target = deepcopy(agent.critic)
    agent.actor_target = deepcopy(agent.actor)
    return agent


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


def save_params(dic, filename: str = 'agent_params'):
    def getValidType(dic):
        dic1 = dict()
        valid_type = ['<class \'int\'>', '<class \'float\'>', '<class \'str\'>', '<class \'NoneType\'>',
                      '<class \'list\'>', '<class \'bool\'>']

        for key, value in dic.items():
            if str(type(value)) in valid_type:
                dic1[key] = value
            if str(type(value)) == '<class \'dict\'>':
                dic1[key] = getValidType(value)

        return dic1

    b = json.dumps(getValidType(dic))
    f2 = open(filename + '.json', 'w')
    f2.write(b)
    f2.close()

def set_mode(model, mode='train'):
    if mode == 'train':
        model.train()
    else:
        model.eval()


def training_plot(figsize=None,
                  row=None,
                  column=None,
                  training_indexs=None,
                  fontsize=None,
                  name=None,
                  if_criteria=False):
    '''

    :param figsize: size of figure
    :param row:  rows of all subplots
    :param column:  columns of all subplots
    :param training_indexs: the suffix of saved file, also means the number of the training results
    :param fontsize: font size of the legend
    :param name: which index wanna to draw
    :param if_criteria:
    :return:
    '''
    if not if_criteria:
        plt.figure(figsize=figsize)
        for i in training_indexs:  # training loop
            results = np.load(f'training_results{i}.npy', allow_pickle=True).item()
            plt.subplot(row, column, i)
            episodes = len(results.keys())
            interval = episodes // 10
            for ep in range(1, episodes + 1, interval):
                plt.title(f'training{i}')
                plt.plot(results[ep][name], label=f'ep{ep}')
            plt.legend(fontsize=fontsize)
        plt.show()
    else:
        plt.figure(figsize=figsize)
        for i in training_indexs:  # training loop
            results = np.load(f'training_results{i}.npy', allow_pickle=True).item()
            plt.subplot(row, column, i)
            criteria_list = []
            episodes = len(results.keys())
            for ep in range(1, episodes + 1):
                criteria_list.append(results[ep][name][0])
            plt.title(f'training{i}')
            plt.plot(criteria_list)

def trans_tensor(data):
    return torch.tensor(data, dtype=torch.float32)