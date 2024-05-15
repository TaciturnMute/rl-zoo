import pandas as pd


class My_YahooDownloader:

    '''
    获取原始数据，并且进行初步整理。
    直接提供需要处理的csv文件的路径。并非利用yahoo finance API获取数据。
    初步整理后的数据形式为：
       date  open  high  low  close  volume  tic  day
    0
    1
    2
    '''
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 data_dir: str,
                 csv_list: list):
        '''
        :param start_date: 待处理数据的起始日期
        :param end_date: 终止日期
        :param data_dir: 成分股原始数据所在路径
        :param csv_list: 所有csv文件名称的列表
        '''
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir  # 所有原始的csv文件的路径
        self.csv_list = csv_list  # 所有csv文件名称

    def fetch_data(self) -> pd.DataFrame:

        data_df = pd.DataFrame()
        for tic in self.csv_list:  # tic的形式为AAPL.csv

            temp_df = pd.read_csv(self.data_dir + tic)  # 路径+文件名。读取文件
            temp_df.Date = pd.to_datetime(temp_df.Date)
            temp_df = temp_df[(temp_df.Date >= pd.to_datetime(self.start_date)) & (temp_df.Date <= pd.to_datetime(self.end_date))]

            temp_df["tic"] = tic[:-4]
            data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index(drop=True)
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")

        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))

        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        '''

        :param df: 初步整理好的数据
        :return: 筛选了一部分股票的数据
        '''
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df

# import os
# from stockstats import StockDataFrame as Sdf
# data_dir='E:/强化学习/强化学习代码/数据集/DOW原始数据/config_tickers.DOW_30_TICKER/'
# csv_list=os.listdir(data_dir)
# df=My_YahooDownloader(start_date='20090101', end_date='20200101',
#                                        data_dir=data_dir, csv_list=csv_list).fetch_data()
# # stock = Sdf.retype(df.copy())
# # print(stock)
# # print(stock[stock.tic == 'AAPL']['macd'])
# # print(df.pivot_table(index="date", columns="tic", values="close").dropna(axis=1))
#
# df_price_pivot = df.pivot(index="date", columns="tic", values="close")
# df_price_pivot = df_price_pivot.pct_change()
# print([x for x in df_price_pivot])
