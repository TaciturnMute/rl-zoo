from __future__ import annotations
import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from typing import Tuple
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl_myself.plot import get_daily_return
from finrl_myself.metrics import *

matplotlib.use("Agg")

'''
this env is used for portfolio management baseline, 
env won't execute softmax to the action.
'''


class StockPortfolioEnv(gym.Env):
    '''
    Pt = sum(portfolio_value_vector * price_pct)
    '''
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df: pd.DataFrame = None,
            stock_dim: int = None,
            hmax: int = None,
            initial_amount: int = None,
            buy_cost_pct: list = None,
            sell_cost_pct: list = None,
            reward_scaling: float = None,
            state_space: int = None,
            action_space: int = None,
            tech_indicator_list: list = None,  # indicators env used
            turbulence_threshold=None,
            lookback: int = 252,
            day: int = 0,
            reward_aliase: str = None,
    ):
        self.reward_aliases = ['asset_diff', 'asset_diff_dji', 'sharpe_ratio_diff']
        self.day = day
        self.lookback = lookback  # ?
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_dim = state_space
        self.action_dim = action_space
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        # self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))  # continuous
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
                                            )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.date = self._get_date()
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.last_weights = np.array([1] * self.stock_dim) / self.stock_dim
        self.portfolio_value = self.initial_amount
        self.portfolio_value_vector = self.portfolio_value * self.last_weights

        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.cost = 0
        self.trades = 0
        self.reward_aliase = reward_aliase
        self.reward = np.nan
        self.date_range = [self.df.date.unique()[0], self.df.date.unique()[-1]]
        if self.reward_aliase == 'asset_diff_dji':
            self._dji_bah()

    def _dji_bah(self):
        self.dji = self.df.drop_duplicates(subset=['date'])[['DJI', 'date']]
        self.dji_shares = self.initial_amount / self.dji.iloc[0]['DJI']
        self.dji_assets = copy.deepcopy(self.dji)
        self.dji_assets.set_index(keys='date', drop=True, inplace=True)
        self.dji_assets['DJI'] = self.dji_assets['DJI'] * self.dji_shares

    def _transaction(self, weights):
        def _sell_asset(index, sell_value) -> float:
            self.portfolio_value -= sell_value
            self.portfolio_value_vector[index] -= sell_value
            # the sold value divided into two parts: Cash and Transaction Cost
            self.cost += sell_value * self.sell_cost_pct[index]  # add sell transaction cost
            cash_obtain_in_this_sell = sell_value * (1 - self.sell_cost_pct[index])  # obtain less
            self.trades += 1
            return cash_obtain_in_this_sell

        def _buy_asset(index, buy_value, cash) -> Tuple[float, bool]:
            # check if the asset is able to buy, cash should > 0
            if_terminal = False
            avaliable_value = cash / (1 + self.buy_cost_pct[index])  # max asset value that can purchase
            if avaliable_value < buy_value:
                buy_value = avaliable_value
                if_terminal = True
            cash -= buy_value * (1 + self.buy_cost_pct[index])
            # cash used is divided into two parts: Transaction cost and Bought Asset Value.
            self.cost += buy_value * self.buy_cost_pct[index]
            self.portfolio_value_vector[index] += buy_value
            self.portfolio_value += buy_value
            self.trades += 1
            return cash, if_terminal

        # assert abs(sum(self.portfolio_value_vector) - self.portfolio_value) < 0.01
        new_portfolio_value_vector = self.portfolio_value * weights

        actions = new_portfolio_value_vector - self.portfolio_value_vector  # get change value

        argsort_actions = np.argsort(actions)
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]  # get buy asset index
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]  # get sell asset index

        # sell first
        cash = 0
        for index in sell_index:
            cash += _sell_asset(index, abs(actions[index]))

        # then buy, the cost can't be larger than cash
        for index in buy_index:
            cash, terminal = _buy_asset(index, abs(actions[index]), cash)
            if terminal:
                break

    def _cal_reward(self):

        # calculate reward
        if self.reward_aliase == 'asset_diff':
            self.reward = (self.portfolio_value - self.last_portfolio) * self.reward_scaling
        elif self.reward_aliase == 'asset_diff_dji':
            self.reward = (self.portfolio_value - self.dji_assets.loc[self.date].values[0]) * self.reward_scaling
        elif self.reward_aliase == 'sharpe_ratio_diff':
            returns = pd.DataFrame(self.asset_memory)
            returns.insert(0,"date", list(self.date_memory))
            returns.dropna()
            returns.columns = ['date', 'account_value']
            if np.isnan(self.reward) or len(self.date_memory) <= 10:
                self.reward = np.random.randn(1)[0] * 2
            else:
                self.reward = (sharpe_ratio(get_daily_return(returns)) - sharpe_ratio(
                    get_daily_return(returns.iloc[:-1]))) * self.reward_scaling
        else:
            assert self.reward_aliase in self.reward_aliases, \
                f"invalid reward type, supported reward types are{self.reward_aliases}"

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        assert len(actions.shape) == 1
        # terminal
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if not self.terminal:

            # save portfolio before action is executed
            self.last_portfolio = self.portfolio_value  # 交易费也是动作带来的影响，所以计入奖励内。

            # get action
            weights = actions

            # adjust the portfolio allocation and consider transaction cost
            self._transaction(weights)

            last_data = self.data  # vt-1

            # state transferring
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.date = self._get_date()
            # self.next_date = self._get_date()
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )

            # calcualte portfolio return
            # individual stocks' return * weight
            # after next state is observed, get Pt
            # update portfolio value
            price_pct_vector = (self.data.close.values / last_data.close.values)  # vt/vt-1
            portfolio_return = sum((price_pct_vector - 1) * weights)
            # new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value_vector = self.portfolio_value_vector * price_pct_vector  # Pt distribution
            self.portfolio_value = sum(self.portfolio_value_vector)  # Pt
            self.last_weights = self.portfolio_value_vector / self.portfolio_value
            # save into memory
            self.actions_memory.append(weights)
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(self.portfolio_value)

            self._cal_reward()

            return self.state, self.reward, self.terminal, {}
        else:
            self.date = self._get_date()
            # self.next_date = None
            self.returns = pd.DataFrame(self.asset_memory)
            self.returns.insert(0, "date", list(self.df.date.unique()))
            self.returns.dropna()
            self.returns.columns = ['date', 'account_value']
            self.returns = get_daily_return(self.returns)  # daily return

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.date = self._get_date()
        self.last_weights = np.array([1] * self.stock_dim) / self.stock_dim
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        self.portfolio_value_vector = self.portfolio_value * self.last_weights
        self.cost = 0
        self.trades = 0
        self.trades += 1
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.reward = np.nan
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date
