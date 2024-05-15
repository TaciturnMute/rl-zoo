from __future__ import annotations
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from collections import deque
from typing import List, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl_myself.plot import get_daily_return
from finrl_myself.metrics import *


class StockTradingEnv_series(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        history: int,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        reward_aliase: str = None,
    ):
        self.reward_aliases = ['asset_diff', 'sharpe_rario_diff']
        self.day = day
        self.history = history
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.state_dim = self.state_space
        self.action_dim = self.action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.date = self._get_date()
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.reward_aliase = reward_aliase

        self.state = self._initiate_state()
        self.state_deque = deque([self.state] * self.history, maxlen=self.history) ###############

        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        self._seed()
        self.reward = np.nan

    def _sell_stock(self, index: int, action: int):
        def _do_sell_normal():
            if (self.state[index + 2 * self.stock_dim + 1] != True):      # stock's macd
              # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                    sell_amount = (self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index]))
                    # update balance
                    self.state[0] += sell_amount
                    # update holding shares
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index])
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]  # price
                            * sell_num_shares  # shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (self.state[index + 2 * self.stock_dim + 1] != True):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]  # price
                    * buy_num_shares  # shares
                    * (1 + self.buy_cost_pct[index])  # add cost
                )
                # ##################修改的地方（添加）
                # self.original_buy_amount = (
                #     self.state[index + 1]  # price
                #     * action  # shares
                #     * (1 + self.buy_cost_pct[index])  # add cost
                # )
                # #####################
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _cal_total_asset(self):
        # get total asset value at present
        total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        return total_asset

    def _cal_reward(self):
        # calculate reward
        if self.reward_aliase == 'asset_diff':
            self.reward = (self.end_total_asset - self.begin_total_asset) * self.reward_scaling
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
        # assert len(actions.shape) == 1
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if not self.terminal:
            actions = actions * self.hmax  # actions [-100, 100]
            actions = actions.astype(int)  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)  # 意义在于符号，所有动作都为-1，意味着全部股票进入sell状态。进入sell之后就会全部售出

            self.begin_total_asset = self._cal_total_asset()  # get total asset before this trade start

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]   # np.ndarray
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])
            self.actions_memory.append(actions)  # actual executed actions

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.date = self._get_date()
            if self.turbulence_threshold is not None:   # update turbulence
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()       # state transfer, get St+1
            # update series states
            self.state_deque.append(self.state)

            self.end_total_asset = self._cal_total_asset()


            self.asset_memory.append(self.end_total_asset)
            self.date_memory.append(self._get_date())
            self.state_memory.append(self.state)

            self._cal_reward()
            self.rewards_memory.append(self.reward)
            state_sequence = np.array([state for state in self.state_deque])
            # 添加的地方
            # print(f'current owned stock shares: {self.state[self.stock_dim + 1: self.stock_dim + 1 + self.stock_dim]}')
            # print(f'actually executed action: {actions}')
            # print(f'sell_obtain is: {sell_total_amount}')
            # print(f'the gap between buy_need and sell_obtain is: {buy_total_amount - sell_total_amount}')

            return state_sequence, self.reward, self.terminal, {}
        else:
            state_sequence = np.array([state for state in self.state_deque])
            self.returns = pd.DataFrame(self.asset_memory)
            self.returns.insert(0,"date", list(self.df.date.unique()))
            self.returns.dropna()
            self.returns.columns = ['date', 'account_value']
            self.returns = get_daily_return(self.returns)    # daily return
            return state_sequence, self.reward, self.terminal, {}

    def reset(self) -> np.ndarray:

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.date = self._get_date()
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        self.reward = np.nan

        # initiate state
        self.state = self._initiate_state()
        self.state_deque = self.state_deque = deque([self.state] * self.history, maxlen=self.history)  #############
        state_sequence = np.array([state for state in self.state_deque])

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        return state_sequence

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self) -> list:
        if self.initial:  #默认是True，没有previous information。
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (                             #初始化状态用到了起始资本、第1天的close price、所有股票的股份（全为0）、第1天的科技指标。
                    [self.initial_amount]
                    + self.data.close.values.tolist() #注意，这里是一个列表，真正的列表，每个元素为close的值，下面的[self.data.close]也是一个列表，但却是只包含一个series作为元素的列表。
                    + self.num_stock_shares           #等价于[0] * self.stock_dim，因为num_stock_shares就是长度为self.stock_dim的零元素列表。
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (                             #四个列表元素构成的元组，第一个列表只有一个数字，第二个列表元素为一个series，第三个是长度为 self.stock_dim 的 0 元素列表，第四个是元素为series的列表。
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self) -> list:
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


