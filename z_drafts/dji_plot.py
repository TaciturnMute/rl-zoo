import pandas as pd
import matplotlib.pyplot as plt
from finrl_myself.constants import *
from finrl_myself.metrics import *
from finrl_myself.plot import get_daily_return, my_backtest_plot

TRAIN_START_DATE = '2006-01-01'
df = pd.read_csv('E:\强化学习\强化学习代码\数据集\DOW原始数据\DJI\DJI.csv')
df = df[['Close','Date']]
df.columns = ['close','date']
df['date'] = pd.to_datetime(df.date)

df_train = df[(df['date'] >=pd.to_datetime(TRAIN_START_DATE)) &
(df['date'] <= pd.to_datetime(TRAIN_END_DATE))
]
df_train_validation = df[(df['date'] >=pd.to_datetime(TRAIN_START_DATE)) &
(df['date'] <= pd.to_datetime(VALIDATE_END_DATE))
]
df_test = df[(df['date'] >=pd.to_datetime(TEST_START_DATE)) &
(df['date'] <= pd.to_datetime(TEST_END_DATE))
]
print(df_train_validation)
df_test['account_value'] = df_test['close'] * 38.85
plt.plot(df_test['account_value'])
plt.show()
returns = get_daily_return(df_test)
print(sharpe_ratio((returns)))
print(annual_return(returns))
print(cum_returns_final(returns))

df_train_validation['account_value'] = df_train_validation['close'] * 92.187
plt.plot(df_train_validation['account_value'])
plt.show()
returns = get_daily_return(df_train_validation)
print(sharpe_ratio((returns)))
print(annual_return(returns))
print(cum_returns_final(returns))

