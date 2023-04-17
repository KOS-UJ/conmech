from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch


class FeaturesStatisticsPandas:
    def __init__(self, label, data, columns=None):
        self.label = label

        self.pandas_data = pd.DataFrame(data.numpy())

        if columns is None:
            columns = [str(i) for i in range(data.shape[1])]
        self.pandas_data.columns = columns

        self.mean = torch.mean(data, axis=0)
        self.std = torch.std(data, axis=0)
        self.max_abs = torch.max(torch.abs(data), axis=0).values

    def describe(self):
        return self.pandas_data.describe()


def aggregate(old_value, old_size, new_data, batch_value):
    batch_size = len(new_data)
    new_size = old_size + batch_size
    if old_size == 0:
        assert old_value is None
        return batch_value

    new_proportion = np.float64(batch_size) / np.float64(new_size)

    # https://gist.github.com/davidbau/00a9b6763a260be8274f6ba22df9a145
    # Chen-style update
    delta = batch_value.sub_(old_value).mul_(new_proportion)
    new_value = old_value + delta

    # comp = (1 - (new_value / old_value).abs()).abs().max()
    # if comp > 0.7:
    #     a = 0
    # print(comp)
    return new_value


def get_mean(old_mean, old_size, new_data):
    batch_mean = new_data.mean(axis=0, dtype=torch.float64)
    return aggregate(
        old_value=old_mean, old_size=old_size, new_data=new_data, batch_value=batch_mean
    )


def get_variance(old_variance, old_size, new_data, true_mean=None):
    if true_mean is None:
        batch_mean = new_data.mean(axis=0, dtype=torch.float64)
        batch_variance = ((new_data - batch_mean) ** 2).sum(axis=0, dtype=torch.float64) / (
            len(new_data) - 1
        )  #  unbiased estimator
    else:
        batch_variance = ((new_data - true_mean) ** 2).mean(axis=0, dtype=torch.float64)
    return aggregate(
        old_value=old_variance,
        old_size=old_size,
        new_data=new_data,
        batch_value=batch_variance,
    )


def get_max_abs(old_max, new_data):
    new_max = new_data.abs().max(axis=0).values
    if old_max is None:
        return new_max
    return torch.max(new_max, old_max)


class FeaturesStatistics:
    def __init__(self):
        self.mean = None
        self.var = None
        self.std = None
        self.max_abs = None
        self.size = 0
        self._mean_ready = False

    def set_mean_and_max_abs(self, new_data):
        self.max_abs = get_max_abs(old_max=self.max_abs, new_data=new_data)
        self.mean = get_mean(old_mean=self.mean, old_size=self.size, new_data=new_data)
        self.size += len(new_data)

    def finalaze_mean(self):
        self._mean_ready = True
        self.size = 0

    def set_variance(self, new_data):
        assert self._mean_ready
        self.var = get_variance(
            old_variance=self.var,
            old_size=self.size,
            new_data=new_data,
            true_mean=self.mean,
        )
        self.size += len(new_data)

    def finalize_variance(self):
        self.std = self.var.sqrt()
