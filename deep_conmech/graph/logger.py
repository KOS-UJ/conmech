import json

import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.profiler import ProfilerActivity, profile
from torch.utils.tensorboard.writer import SummaryWriter

from deep_conmech.data import base_dataset
from deep_conmech.data.dataset_statistics import FeaturesStatistics
from deep_conmech.training_config import TrainingConfig


class Logger:
    def __init__(
        self,
        dataset: base_dataset.BaseDataset,
        config: TrainingConfig,
    ):
        self.dataset = dataset
        self.config = config
        self.writer = SummaryWriter(self.current_log_catalog)

    def save_parameters_and_statistics(self):
        print("Saving parameters...")
        self.save_parameters()
        for layer_number in [1]:
            if self.config.log_dataset_stats:
                print(f"Saving statistics (layer {layer_number})...")
                statistics = self.dataset.get_statistics(layer_number=layer_number)
                self.save_hist_and_json(
                    st=statistics.nodes_statistics,
                    columns=self.config.td.dimension + 1,
                    name=f"nodes_statistics_layer{layer_number}",
                )
                self.save_hist_and_json(
                    st=statistics.edges_statistics,
                    columns=self.config.td.dimension + 1,
                    name=f"edges_statistics_layer{layer_number}",
                )
                self.save_hist_and_json(
                    st=statistics.target_statistics,
                    columns=self.config.td.dimension + 1,
                    name=f"target_statistics_layer{layer_number}",
                )

    def save_parameters(self):
        def pretty_json(value):
            dictionary = vars(value)
            json_str = json.dumps(dictionary, indent=2)
            return "".join("\t" + line for line in json_str.splitlines(True))

        data_str = pretty_json(self.config.td)
        self.writer.add_text(f"{self.config.current_time}_parameters.txt", data_str, global_step=0)
        file_path = f"{self.current_log_catalog}/parameters_{self.dataset.data_id}.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(data_str)

    def save_hist_and_json(self, st: FeaturesStatistics, columns: int, name: str):
        df = st.pandas_data
        self.save_hist(df=df, columns=columns, name=name)

        # normalized_df = (df - df.mean()) / df.std()
        # self.save_hist(df=normalized_df, name=f"{name}_normalized")

        data_str = st.describe().to_json()
        self.writer.add_text(f"{self.config.current_time}_{name}.txt", data_str, global_step=0)

    def save_hist(self, df: DataFrame, columns: int, name: str):
        # pandas_axs = st.pandas_data.hist(figsize=(20, 10))  # , ec="k")
        scale = 7
        rows = (df.columns.size // columns) + df.columns.size % columns
        fig, axs = plt.subplots(
            rows, columns, figsize=(columns * scale, rows * scale), sharex="row", sharey="row"
        )  # , sharex="col", sharey="row"
        for i in range(rows * columns):
            row, col = i // columns, i % columns
            if i < df.columns.size:
                df.hist(
                    column=df.columns[i], bins=100, ax=axs[row, col]
                )  # bins=12 , figsize=(20, 18)
            else:
                axs[row, col].axis("off")

        fig.tight_layout()
        fig.savefig(f"{self.current_log_catalog}/hist_{name}.png")

    @property
    def current_log_catalog(self):
        return f"{self.config.log_catalog}/{self.config.current_time}"

    def get_and_start_profiler(self):
        def trace_handler(prof):
            print("Saving profiler raport...")
            # output = prof.key_averages().table(row_limit=10, sort_by="cpu_time_total")
            # print(output)
            # prof.export_chrome_trace(f"./log/profiler_trace.json")
            # prof.export_stacks("profiler_stacks_{prof.step_num}.txt", "self_cuda_time_total")
            torch.profiler.tensorboard_trace_handler(self.current_log_catalog)(prof)

        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            # record_shapes=True,
            # with_stack=True,
            schedule=torch.profiler.schedule(skip_first=2, wait=0, warmup=2, active=6, repeat=1),
            on_trace_ready=trace_handler,
        )
        profiler.start()
        return profiler
