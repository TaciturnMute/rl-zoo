from stockstats import StockDataFrame as Sdf
from finrl_myself.data_preprocessor.my_yahoodownloader import My_YahooDownloader
from typing import List
import pandas as pd
import numpy as np

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

class My_FeatureEngineer:
    '''
    主要使用preprocess_data函数，处理初步整理后的数据。获得相应指标，并且清洗数据。
    获得的指标分三部分：finrl 默认的indicators，用户自定义的indicators，turbulence。
    '''

    def __init__(
            self,
            tech_indicator_list: List = INDICATORS,
            use_technical_indicator: bool = True,
            use_turbulence=False,
            user_defined_feature=False,
            use_vix=False,
            vix_data_dir=None,
            csv_list=None
    ):
        '''

        :param use_technical_indicator: 是否使用技术指标
        :param tech_indicator_list: 技术指标种类列表
        :param use_vix: 是否使用vix(Volatility index)波动率指数，反映市场状况的指标
        :param use_turbulence: 是否使用turbulence（用来监控市场风险）
        :param user_defined_feature:

        :param vix_data_dir: vix指标的csv文件所在路径
        :param csv_list: vix指标的csv文件名
        '''
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
        self.vix_data_dir = vix_data_dir
        self.csv_list = csv_list

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df, self.vix_data_dir, self.csv_list)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def clean_data(self, data):
        """
        将处于同一天的数据的索引设置为相同值。删去存在缺失的股票。
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]   # 将处于同一天的数据的索引设置为同一值
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)  # 删除列, DOW这支成分股缺失严重，所以将其剔除。
        tics = merged_closes.columns  # 获取剩余股票代码
        df = df[df.tic.isin(tics)]  # 获取剩余股票代码的数据
        # df = data.copy()
        # list_ticker = df["tic"].unique().tolist()
        # only apply to daily level data, need to fix for minute level
        # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        # combination = list(itertools.product(list_date,list_ticker))

        # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        # df_full = df_full[df_full['date'].isin(df['date'])]
        # df_full = df_full.sort_values(['date','tic'])
        # df_full = df_full.fillna(0)
        return df

    def add_technical_indicator(self, data):
        """
        计算所有的股票的indicators。
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):  # 计算所有成分股对应的indicator
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]  # 利用stockstats API计算indicator
                    temp_indicator = pd.DataFrame(temp_indicator)  # series变为dataframe
                    temp_indicator["tic"] = unique_ticker[i]  # 赋tic
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][  # 赋date
                        "date"
                    ].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            # 将计算结果合并到df中去
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def add_user_defined_feature(self, data):
        """
        如果希望添加其他指标，可以修改该函数。
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    def add_vix(self, data, vix_data_dir, csv_list):
        """
        给出vix的原始数据路径，合并vix数据。
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()

        df_vix = My_YahooDownloader(start_date=df.date.min(), end_date=df.date.max(), data_dir=vix_data_dir,
                                    csv_list=csv_list).fetch_data()

        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        # use returns to calculate turbulence，使用returns计算turbulence
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        df_price_pivot = df_price_pivot.pct_change()  # 所有股票当天相较前一天(上一条)的close数据比例。
        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start # 开始的一年都是0。
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            # 使用当前日期过去一年的滑动数据
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
                ]

            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                                  hist_price.isna().sum().min():
                                  ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()  # 计算剩余数据的协方差
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )   # 当前日期股票价格减去过去一年股票价格均值，即(yt-u)
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:  # turbulence要被记录。
                count += 1
                if count > 2:   # 避免异常点，所以过两次再计入
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df