import numpy as np
import pandas as pd
from typing import List,Tuple


class MACDStrategy():

    def __init__(self, trend_combinations: List[Tuple[float, float]] = None):
        """Used to calculated the combined MACD signal for a multiple short/signal combinations,
        as described in https://arxiv.org/pdf/1904.04912.pdf
        Args:
            trend_combinations (List[Tuple[float, float]], optional): short/long trend combinations. Defaults to None.
        """
        if trend_combinations is None:
            self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trend_combinations = trend_combinations

    @staticmethod
    def calc_signal(srs: pd.Series, short_timescale: int, long_timescale: int) -> float:
        """Calculate MACD signal for a signal short/long timescale combination
        Args:
            srs ([type]): series of prices
            short_timescale ([type]): short timescale
            long_timescale ([type]): long timescale
        Returns:
            float: MACD signal
        """

        def _calc_halflife(timescale):
            return np.log(0.5) / np.log(1 - 1 / timescale)

        macd = (
            srs.ewm(halflife=_calc_halflife(short_timescale)).mean()      #adjust=True
            - srs.ewm(halflife=_calc_halflife(long_timescale)).mean()
        )
        q = macd / srs.rolling(63).std().fillna(method="bfill")   #normalized
        return q / q.rolling(252).std().fillna(method="bfill")     #normalized

    @staticmethod
    def scale_signal(y):
        return y * np.exp(-(y ** 2) / 4) / 0.89

    def calc_combined_signal(self, srs: pd.Series) -> float:
        """Combined MACD signal
        Args:
            srs (pd.Series): series of prices
        Returns:
            float: MACD combined signal
        """
        return np.sum(
            [self.calc_signal(srs, S, L) for S, L in self.trend_combinations]
        , axis=0) / len(self.trend_combinations)