from dataclasses import dataclass
from typing import List

import pandas as pd
import torch


class FeaturesStatistics:
    def __init__(self, label, data, columns=None):
        self.label = label
        self.pandas_data = pd.DataFrame(data.numpy())

        if columns is None:
            columns = [str(i) for i in range(data.shape[1])]
        self.pandas_data.columns = columns

        self.data_mean = torch.mean(data, axis=0)
        self.data_std = torch.std(data, axis=0)

    def describe(self):
        return self.pandas_data.describe()

    # @property
    # def columns_count(self):
    #     return len(self.pandas_data.columns)


@dataclass
class DatasetStatistics:
    data: List[FeaturesStatistics]

    def normalize(self, layer_list):
        def test_and_set(value, id, label):
            assert self.data[id].label == label
            if value.device != self.data[id].data_std.device:
                self.data[id].data_std = self.data[id].data_std.to(value.device)
            return torch.nan_to_num(value / self.data[id].data_std)

        layer_list[1].x = test_and_set(layer_list[1].x, 0, "sparse_nodes")
        layer_list[1].edge_attr = test_and_set(layer_list[1].edge_attr, 1, "sparse_edges")
        layer_list[1].edge_attr_to_down = test_and_set(
            layer_list[1].edge_attr_to_down, 2, "multilayer_edges"
        )
        layer_list[0].x = test_and_set(layer_list[0].x, 3, "dense_nodes")
        layer_list[0].edge_attr = test_and_set(layer_list[0].edge_attr, 4, "dense_edges")

        # Sparse
        layer_list[0].edge_attr[:, -4:] = 0.0  # forces # shape = 4

        # Dense
        layer_list[1].edge_attr[:, -4:] = 0.0  # forces # shape = 16

        # Multilayer edges
        layer_list[1].edge_attr_to_down[:, -4:] = 0.0  # forces

        return layer_list
