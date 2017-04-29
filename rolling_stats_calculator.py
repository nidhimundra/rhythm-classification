"""
 Created by Nidhi Mundra on 26/04/17.
"""

import pandas as pd
import math


class RollingStatsCalculator:
    def __init__(self):
        pass

    def rolling_mean(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_mean(data, window=window)[0::window])

    def rolling_count(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_count(data, window=window)[0::window])

    def rolling_kurt(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_kurt(data, window=window)[0::window])

    def rolling_max(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_max(data, window=window)[0::window])

    def rolling_min(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_min(data, window=window)[0::window])

    def rolling_skew(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_skew(data, window=window)[0::window])

    def rolling_sum(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_sum(data, window=window)[0::window])

    def rolling_var(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_var(data, window=window)[0::window])

    def rolling_std(self, data, window=10):
        return self.append_first_valid_occurence(pd.rolling_std(data, window=window)[0::window])

    @staticmethod
    def append_first_valid_occurence(output):
        value = next(x for x in output if not math.isnan(x))
        return [value if math.isnan(x) else x for x in output]
