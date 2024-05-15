from finrl_myself.env.env_stocktrading import StockTradingEnv
from finrl_myself.ddpg.agent import DDPG
from finrl_myself.constants import *
from finrl_myself.data_preprocessor.data_generate import portfolio_data_generate,stock_trade_data_generate
from finrl_myself.utils import data_split
from finrl_myself.ddpg.kwargs import *
from finrl_myself.utils import save_params
from finrl_myself.test import test
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'E:/强化学习/强化学习代码/数据集/DOW原始数据/config_tickers.DOW_30_TICKER/'
vix_data_dir = 'E:/强化学习/强化学习代码/数据集/DOW原始数据/^VIX/'
TRAIN_START_DATE = '2010-01-01'
df = stock_trade_data_generate(
    data_dir=data_dir,
    start_date=TRAIN_START_DATE,
    end_date=TEST_END_DATE,
    use_technical_indicator=True,
    use_turbulence=True,
    user_defined_feature=False,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    vix_data_dir=vix_data_dir,
)
df_train = data_split(df,TRAIN_START_DATE,TRAIN_END_DATE)
df_validation = data_split(df,VALIDATE_START_DATE,VALIDATE_END_DATE)
df_test = data_split(df,TEST_START_DATE,TEST_END_DATE)
df_train_validation = data_split(df,TRAIN_START_DATE,VALIDATE_END_DATE)

validation_risk_indicator = df_validation.drop_duplicates(subset = ['date'])
validation_vix_threshold = validation_risk_indicator.vix.quantile(0.996)
test_risk_indicator = df_test.drop_duplicates(subset = ['date'])
test_vix_threshold = test_risk_indicator.vix.quantile(0.996)

STOCK_DIM = len(df_train.tic.unique())
ACTION_DIM = STOCK_DIM  # 28
STATE_DIM = 1 + ACTION_DIM * (1+1) + ACTION_DIM * 8

env_trade_kwargs.update({
    'num_stock_shares':STOCK_DIM*[0],
    'stock_dim':STOCK_DIM,
    'state_space':STATE_DIM,
    'action_space':STOCK_DIM,
    'buy_cost_pct':[0.001]*ACTION_DIM,
    'sell_cost_pct':[0.001]*ACTION_DIM,
    'reward_aliase':'asset_diff',
    'reward_scaling':1e-4,
})
env_train = StockTradingEnv(df_train, **env_trade_kwargs)
env_train_validation = StockTradingEnv(df_train_validation, **env_trade_kwargs)

env_validation = StockTradingEnv(df_validation, **env_trade_kwargs)
env_validation.risk_indicator_col = 'vix'    # 设置预警
env_validation.turbulence_threshold = validation_vix_threshold
env_test = StockTradingEnv(df_test, **env_trade_kwargs)
env_test.risk_indicator_col = 'vix'           # 设置预警
env_test.turbulence_threshold = test_vix_threshold

ou_noise_kwargs.update({'mu':np.array([0]*ACTION_DIM)})

feature_extractor = 'flatten'
feature_extractor_kwargs = feature_extractors_kwargs[feature_extractor]
feature_extractor_kwargs.update({'observation_space':env_train.observation_space})

actor_kwargs.update({
    'net_arch':[400,300],
    'lr':1e-3,
    'state_dim':STATE_DIM,
    'action_dim':ACTION_DIM,
})
actor_kwargs.update({'feature_extractor_kwargs':feature_extractor_kwargs})

critic_kwargs.update({
    'net_arch': [{'vf': [64, 32], 'qf': [32]}, 16, 16],
    'lr': 1e-3,
    'state_dim': STATE_DIM,
    'action_dim': ACTION_DIM,

})
critic_kwargs['feature_extractor_kwargs'] = feature_extractor_kwargs

agent_kwargs.update({
    'env_train':env_train_validation,
    'env_validation':None,
    'env_test':env_test,
    'actor_aliase':'SequentialMlp',
    'critic_aliase':'DualingFore',
    'critic_kwargs':critic_kwargs,
    'actor_kwargs':actor_kwargs,
    'noise_kwargs':ou_noise_kwargs,
    'if_smooth_noise':False,
})

agent = DDPG(**agent_kwargs)
agent.train()
agent.save('1')