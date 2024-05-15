import torch
import pandas as pd
import datetime
import numpy as np
import matplotlib
import os
from finrl_myself.plot import get_daily_return,backtest_stats,my_get_baseline,my_backtest_plot

matplotlib.use("Agg")

class test():
    '''
    in sample test\out of sample test\backtesting\backtesting plot
    '''

    def __init__(
            self,
            agent,
            env_train,
            env_test,
            risk_free: float = 0.03,
            baseline_data_dir: str = None,
            TRAIN_START_DATE: str = None,
            TRAIN_END_DATE: str = None,
            TEST_START_DATE: str = None,
            TEST_END_DATE: str = None,
            if_actor:bool = None,
    ):
        self.env_train = env_train
        self.env_test = env_test
        self.agent = agent
        self.risk_free = risk_free
        self.baseline_data_dir = baseline_data_dir
        self.TRAIN_START_DATE = TRAIN_START_DATE
        self.TRAIN_END_DATE = TRAIN_END_DATE
        self.TEST_START_DATE = TEST_START_DATE
        self.TEST_END_DATE = TEST_END_DATE
        self.if_actor = if_actor
        self.env_action_range = [-1,1]
        # set test mode, otherwise results is unstable
        if self.if_actor:
            print('inference mode! ')
            self.agent.actor.eval()
        else:
            print('inference mode! ')
            self.agent.policy.eval()


    def reset_train(self):
        self.env_train.reset()

    def reset_trade(self):
        self.env_test.reset()

    def in_sample_test(self):
        print('====================== In Sample Test ======================')
        self.reset_train()
        s = self.env_train.reset()  # list
        done = False
        while not done:
            s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,) + s.shape)
            with torch.no_grad():
                if self.if_actor:
                    a = self.agent.actor.get_actions(s_tensor, deterministic=True).detach().numpy().reshape(-1)  # tensor
                else:
                    a = self.agent.policy.get_actions(s_tensor, deterministic=True).detach().numpy().reshape(-1)

            s_, r, done, _ = self.env_train.step(a)  # list,scalar,bool,empty_dict
            s = s_
        print(f'the last ten timesteps\' assets of in sample test are {self.env_train.asset_memory[-10:]}')
        # plt.figure(figsize=(6, 3))
        # plt.plot(self.env_train.asset_memory)

    def out_of_sample_test(self):
        print('====================== Out of Sample Test ======================')
        self.reset_trade()
        s = self.env_test.reset()  # list
        done = False
        while not done:
            s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,)+s.shape)
            with torch.no_grad():
                if self.if_actor:
                    a = self.agent.actor.get_actions(s_tensor, deterministic=True).detach().numpy().reshape(-1)  # tensor
                else:
                    a = self.agent.policy.get_actions(s_tensor, deterministic=True).detach().numpy().reshape(-1)
            s_, r, done, _ = self.env_test.step(a)  # list,scalar,bool,empty_dict
            s = s_

        print(f'the last ten timesteps\' assets of out of sample test are {self.env_test.asset_memory[-10:]}')
        # plt.figure(figsize=(6, 3))
        # plt.plot(self.env_gym_trade.asset_memory)

    def backtesting_test(self):
        self.df_account_value = pd.DataFrame(self.env_test.asset_memory)
        self.df_account_value.columns = ['account_value']
        self.df_account_value.insert(0, "date", np.array(self.env_test.df.date.unique()))

        print("====================== Get Test Backtest Results ======================")
        now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
        self.perf_stats_all = backtest_stats(account_value=self.df_account_value, value_col_name='account_value')
        self.perf_stats_all = pd.DataFrame(self.perf_stats_all)
        self.dr_test_trade = get_daily_return(self.df_account_value)

    def backtesting_train(self):
        self.df_account_value_train = pd.DataFrame(self.env_train.asset_memory)
        self.df_account_value_train.columns = ['account_value']
        self.df_account_value_train.insert(0, "date", np.array(self.env_train.df.date.unique()))
        self.dr_test_train = get_daily_return(self.df_account_value_train)

        print("====================== Get Train Backtest Results ======================")
        now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
        self.perf_stats_all_train = backtest_stats(account_value=self.df_account_value_train,
                                                   value_col_name='account_value')
        self.perf_stats_all_train = pd.DataFrame(self.perf_stats_all_train)

    def calculate_sharpe_ratio_with_risk_free(self):
        risk_free_daily = self.risk_free / 365
        sharp_ratio_adj_trade = (np.nanmean(self.dr_test_trade - risk_free_daily) / np.nanstd(
            self.dr_test_trade - risk_free_daily)) * np.sqrt(252)
        sharp_ratio_adj_train = (np.nanmean(self.dr_test_train - risk_free_daily) / np.nanstd(
            self.dr_test_train - risk_free_daily)) * np.sqrt(252)
        return sharp_ratio_adj_trade, sharp_ratio_adj_train

    def log(self):
        '''
        after in sample test and out of sample test, do the log
        '''
        # assets results
        self.df_assets_results = pd.concat([pd.Series(self.env_train.date_memory[::-1]),
                                            pd.Series(self.env_train.asset_memory[::-1]),
                                            pd.Series(self.env_test.date_memory[::-1]),
                                            pd.Series(self.env_test.asset_memory[::-1])], axis=1)
        self.df_assets_results.columns = ['train_date', 'train_asset', 'test_date', 'test_asset']

        # criteria results
        self.df_criteria = pd.concat([self.perf_stats_all[0],
                                      self.perf_stats_all_train[0]], axis=1)
        self.df_criteria.columns = ['test_results', 'train_results']

        # other results
        _sharp_ratio_adj_trade, _shapre_ratio_adj_train = self.calculate_sharpe_ratio_with_risk_free()
        other_results_index = ['sharpe_ratio_adj', 'cost', 'trades_number']
        other_results_train = [_shapre_ratio_adj_train,
                               self.env_train.cost,
                               self.env_train.trades]
        other_results_trade = [_sharp_ratio_adj_trade,
                               self.env_test.cost,
                               self.env_test.trades]
        self.df_other_results = pd.concat([pd.Series(other_results_trade),
                                           pd.Series(other_results_train)], axis=1)
        self.df_other_results.index = pd.Series(other_results_index)
        self.df_other_results.columns = ['test_results', 'train_results']

        # summary，merge all results in a dataframe, one model results
        self.df_results = pd.concat([self.df_criteria.append(self.df_other_results).reset_index(),
                                     self.df_assets_results
                                     ], axis=1)
        # print(self.df_results.iloc[:20])

    def get_baselines(self):
        self.baseline_csv_list = os.listdir(self.baseline_data_dir)
        self.baseline_df = my_get_baseline(data_dir=self.baseline_data_dir,
                                           start_date=self.TEST_START_DATE,
                                           end_date=self.TEST_END_DATE,
                                           csv_list=self.baseline_csv_list)
        print(f'====================== Get Baseline Stats ======================')
        stats = backtest_stats(self.baseline_df, value_col_name='close')

    def baseline_plot(self):

        print("====================== Compare to DJIA ======================")
        # S&P 500: ^GSPC
        # Dow Jones Index: ^DJI
        # NASDAQ 100: ^NDX
        my_backtest_plot(self.df_account_value,  # trade assets
                         data_dir=self.baseline_data_dir,  # compare with the baseline data result
                         baseline_start=self.TEST_START_DATE,
                         baseline_end=self.TEST_END_DATE,
                         )

    def test_all(self):
        self.in_sample_test()
        self.out_of_sample_test()
        self.backtesting_test()
        self.backtesting_train()
        self.log()
