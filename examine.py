from finrl_myself.utils import set_mode
from finrl_myself.metrics import *
import torch
import numpy as np

# do validation dataset inference and test dataset inference

def validate(env, action_provider, logger):
    if env is None:
        return
    set_mode(action_provider, 'test')
    s = env.reset()  # list
    done = False
    while not done:
        s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,) + s.shape)
        a = action_provider.get_actions(s_tensor, deterministic=True).detach().numpy().reshape(-1)  # (action_dim,)
        s_, r, done, _ = env.step(a)  # ndarray,float,bool,dict
        s = s_
    returns = env.returns
    total_asset_ = np.mean(env.asset_memory)
    cummulative_returns_ = cum_returns_final(returns)
    annual_return_ = annual_return(returns)
    sharpe_ratio_ = sharpe_ratio(returns)
    max_drawdown_ = max_drawdown(returns)
    print('++++++++++++++ validation result +++++++++++++++')
    print(f'validation date range: {env.date_range[0]} -- {env.date_range[1]}')
    print(f'Total asset: {total_asset_}')
    print(f'Cumulative returns: {cummulative_returns_}')
    print(f'Annual return: {annual_return_}')
    print(f'Sharpe ratio: {sharpe_ratio_}')
    print(f'Max drawdown: {max_drawdown_}')
    print('++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.record(val_total_asset=total_asset_,
                       val_cumulative_return=cummulative_returns_,
                       val_annual_return=annual_return_,
                       val_sharpe_ratio=sharpe_ratio_,
                       val_max_drawdown=max_drawdown_)
    set_mode(action_provider, 'train')


def test(env, action_provider, logger):
    if env is None:
        return
    set_mode(action_provider, 'test')
    s = env.reset()  # list
    done = False
    while not done:
        s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,) + s.shape)
        a = action_provider.get_actions(s_tensor, deterministic=True).detach().numpy().reshape(-1)  # (action_dim,)
        s_, r, done, _ = env.step(a)  # ndarray,float,bool,dict
        s = s_
    returns = env.returns
    total_asset_ = np.mean(env.asset_memory)
    cummulative_returns_ = cum_returns_final(returns)
    annual_return_ = annual_return(returns)
    sharpe_ratio_ = sharpe_ratio(returns)
    max_drawdown_ = max_drawdown(returns)
    print('++++++++++++++ test result +++++++++++++++')
    print(f'test date range: {env.date_range[0]} -- {env.date_range[1]}')
    print(f'Total asset: {total_asset_}')
    print(f'Cumulative returns: {cummulative_returns_}')
    print(f'Annual return: {annual_return_}')
    print(f'Sharpe ratio: {sharpe_ratio_}')
    print(f'Max drawdown: {max_drawdown_}')
    print('+++++++++++++++++++++++++++++++++++++++++++')
    logger.record(test_total_asset=total_asset_,
                  test_cumulative_return=cummulative_returns_,
                  test_annual_return=annual_return_,
                  test_sharpe_ratio=sharpe_ratio_,
                  test_max_drawdown=max_drawdown_)
    set_mode(action_provider, 'train')
