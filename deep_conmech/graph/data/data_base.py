import copy
import re
from os import listdir
from os.path import isfile, join

import deep_conmech.common.config as config
import deep_conmech.common.trh as trh
import numpy as np
import deep_conmech.common.plotter.plotter_mapper as plotter_mapper
import torch
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_forces import *
from torch_geometric.loader import DataLoader


def get_print_dataloader(dataset):
    return get_dataloader(dataset, config.BATCH_SIZE, num_workers=0, shuffle=False)


def get_fast_dataloader(dataset):
    return get_dataloader(
        dataset, config.BATCH_SIZE, num_workers=config.DATALOADER_WORKERS, shuffle=True
    )


def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  ############################
    )


def is_memory_overflow(step_tqdm, tqdm_description):
    memory_usage = thh.get_used_memory_gb()
    step_tqdm.set_description(
        f"{tqdm_description} - memory usage {memory_usage:.2f}/{config.GENERATION_MEMORY_LIMIT_GB}"
    )
    return memory_usage > config.GENERATION_MEMORY_LIMIT_GB


def get_procss_data_range(process_id, data_part_count):
    return range(process_id * data_part_count, (process_id + 1) * data_part_count)


def get_indices(path):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    indices = [int(re.sub("[^0-9]", "", filename)) for filename in filenames]
    return indices


def get_process_indices_to_do(process_id, data_part_count, path):
    assigned_data_range = get_procss_data_range(process_id, data_part_count)
    return get_indices_to_do(assigned_data_range, path)


def get_indices_to_do(data_range, path):
    indices_done = get_indices(path)
    indices_to_do = list(set(data_range) - set(indices_done))
    return indices_to_do


def get_assigned_scenarios(all_scenarios, num_workers, process_id):
    scenarios_count = len(all_scenarios)
    if scenarios_count % num_workers != 0:
        raise Exception("Cannot divide data generation work")
    assigned_scenarios_count = int(scenarios_count / num_workers)
    assigned_scenarios = all_scenarios[
        process_id
        * assigned_scenarios_count : (process_id + 1)
        * assigned_scenarios_count
    ]
    return assigned_scenarios


class BaseDatasetDynamic:
    def __init__(self, relative_path, data_count, randomize_at_load):
        self.relative_path = relative_path
        self.data_count = data_count
        self.randomize_at_load = randomize_at_load
        thh.create_folders(self.path)
        thh.create_folders(self.data_path)
        thh.create_folders(self.images_path)

    def generate_all_data(self):
        pass

    @property
    def path(self):
        return f"./datasets/{config.DATA_FOLDER}/{self.relative_path}"

    @property
    def images_path(self):
        return f"{self.path}/images"

    @property
    def data_path(self):
        return f"{self.path}/data"

    def get_setting_path(self, index):
        return f"{self.data_path}/setting_{index}.pt"

    def save(self, setting, exact_normalized_a_torch, index):
        setting_copy = copy.deepcopy(setting)
        setting_copy.exact_normalized_a_torch = exact_normalized_a_torch
        setting_copy.clear_save()
        torch.save(setting_copy, self.get_setting_path(index))

    def load(self, index):
        return torch.load(self.get_setting_path(index))

    def get_example(self, index):
        setting = self.load(index)
        if self.randomize_at_load:
            setting.set_randomization(True)
            exact_normalized_a_torch = Calculator.clean(
                setting, setting.exact_normalized_a_torch
            )
        else:
            exact_normalized_a_torch = setting.exact_normalized_a_torch

        setting.exact_normalized_a_torch = None
        data = setting.get_data(
            f"{thh.get_timestamp()} - {index}", exact_normalized_a_torch
        )
        return data

    def check_and_print(
        self, start_index, current_index, setting, step_tqdm, tqdm_description
    ):
        cutoff = config.PRINT_DATA_CUTOFF
        relative_index = current_index - start_index
        if relative_index < cutoff:
            step_tqdm.set_description(
                f"{tqdm_description} - printing data {current_index} ({relative_index+1}/{cutoff})"
            )
            plotter_mapper.print_setting(setting, current_index, self.images_path)
        if current_index == cutoff:
            step_tqdm.set_description(tqdm_description)

    def __getitem__(self, index):
        return self.get_example(index)

    def __len__(self):
        return self.data_count

    def initialize_data(self):
        thh.create_folders(self.path)
        indices_to_do = get_indices_to_do(range(self.data_count), self.data_path)
        # current_data_count = 0 if len(indices) < 1 else max(indices) + 1
        if not indices_to_do:
            print(f"Taking prepared {self.relative_path} data")
        else:
            print(f"Missing {self.relative_path} data")
            self.generate_all_data()

