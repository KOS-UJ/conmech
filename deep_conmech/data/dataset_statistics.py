from dataclasses import dataclass

import pandas as pd
import torch


class FeaturesStatistics:
    def __init__(self, data, descriprion):
        self.pandas_data = pd.DataFrame(data.numpy())
        self.pandas_data.columns = descriprion

        self.data_mean = torch.mean(data, axis=0)
        self.data_std = torch.std(data, axis=0)

    def describe(self):
        return self.pandas_data.describe()


@dataclass
class DatasetStatistics:
    nodes_statistics: FeaturesStatistics
    edges_statistics: FeaturesStatistics
    target_statistics: FeaturesStatistics
