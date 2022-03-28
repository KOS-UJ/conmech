import copy
import re
from os import listdir
from os.path import isfile, join

import deep_conmech.common.config as config
import deep_conmech.common.plotter.plotter_mapper as plotter_mapper
import numpy as np
import pandas as pd
import torch
from conmech.helpers import cmh, mph
from deep_conmech.graph.helpers import dch, thh
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.solver import Solver
from deep_conmech.simulator.setting.setting_forces import *
from torch_geometric.loader import DataLoader


def print_dataset(dataset, cutoff, timestamp, description):
    print(f"Printing dataset {description}...")
    dataloader = get_print_dataloader(dataset)
    batch = next(iter(dataloader))
    iterations = np.min([len(batch), cutoff])
    #for i in range(iterations):
    #    plotter_mapper.plot_data_setting(batch.setting[i], i, timestamp)

        # for _ in range(100):
        #    setting.set_forces(np.random.uniform(
        #        low= -config.FORCES_DATA_SCALE,
        #        high= config.FORCES_DATA_SCALE,
        #        size=(setting.nodes_count, dim)
        #    ))
        #    test_setting(setting)
        #    a = setting.calculate_normalized()
        #    setting.iterate(a)
        # break


def get_print_dataloader(dataset):
    return get_dataloader(dataset, config.BATCH_SIZE, num_workers=0, shuffle=False)


def get_valid_dataloader(dataset):
    return get_dataloader(
        dataset,
        config.VALID_BATCH_SIZE,
        num_workers=config.DATALOADER_WORKERS,
        shuffle=False,
    )


def get_train_dataloader(dataset):
    return get_dataloader(
        dataset, config.BATCH_SIZE, num_workers=config.DATALOADER_WORKERS, shuffle=True
    )


def get_all_dataloader(dataset):
    return get_dataloader(dataset, len(dataset), num_workers=0, shuffle=False)


def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  ############################
    )


def is_memory_overflow(step_tqdm, tqdm_description):
    memory_usage = dch.get_used_memory_gb()
    step_tqdm.set_description(
        f"{tqdm_description} - memory usage {memory_usage:.2f}/{config.GENERATION_MEMORY_LIMIT_GB}"
    )
    memory_overflow = memory_usage > config.GENERATION_MEMORY_LIMIT_GB
    if memory_overflow:
        step_tqdm.set_description(f"{step_tqdm.desc} - memory overflow")
    return memory_usage > config.GENERATION_MEMORY_LIMIT_GB


def get_process_data_range(process_id, data_part_count):
    return range(process_id * data_part_count, (process_id + 1) * data_part_count)


def get_indices(path):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    indices = [int(re.sub("[^0-9]", "", filename)) for filename in filenames]
    return indices


def get_indices_to_do(data_range, path):
    indices_done = get_indices(path)
    indices_to_do = list(set(data_range) - set(indices_done))
    return indices_to_do


def get_and_check_indices_to_do(data_range, path, process_id):
    indices_to_do = get_indices_to_do(data_range, path)
    if not indices_to_do:
        thh.get_tqdm(range(1), desc=f"Process {process_id} - done", position=process_id)
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


class DatasetStatistics:
    def __init__(self, data, descriprion):
        self.pandas_data = pd.DataFrame(data.numpy())
        self.pandas_data.columns = descriprion

        self.data_mean = torch.mean(data, axis=0)
        self.data_std = torch.std(data, axis=0)

    def describe(self):
        return self.pandas_data.describe()


class BaseDatasetDynamic:
    def __init__(
        self, dimension, relative_path, data_count, randomize_at_load, num_workers
    ):
        self.dimension = dimension
        self.relative_path = relative_path
        self.data_count = data_count
        self.randomize_at_load = randomize_at_load
        self.num_workers = num_workers

    def get_setting_input(self, scenario: Scenario):
        setting = SettingInput(
            mesh_data=scenario.mesh_data,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            create_in_subprocess=False,  #####
        )
        setting.set_randomization(False)
        setting.set_obstacles(scenario.obstacles)
        return setting

    def get_statistics(self):
        dataloader = get_train_dataloader(self)

        nodes_data = torch.empty((0, SettingInput.nodes_data_dim()))
        edges_data = torch.empty((0, SettingInput.edges_data_dim()))
        for data in cmh.get_tqdm(dataloader, desc="Calculating dataset statistics"):
            nodes_data = torch.cat((nodes_data, data.x))
            edges_data = torch.cat((edges_data, data.edge_attr))

        return (
            DatasetStatistics(
                nodes_data, SettingInput.get_nodes_data_description(self.dimension)
            ),
            DatasetStatistics(
                edges_data, SettingInput.get_edges_data_description(self.dimension)
            ),
        )

    def update_data(self):
        pass

    def clear_and_initialize_data(self):
        print(f"Clearing {self.relative_path} data")
        cmh.clear_folder(self.images_path)
        cmh.clear_folder(self.data_path)
        cmh.clear_folder(self.path)
        self.initialize_data()

    def initialize_data(self):
        cmh.create_folders(self.path)
        cmh.create_folders(self.data_path)
        cmh.create_folders(self.images_path)

        indices_to_do = get_indices_to_do(range(self.data_count), self.data_path)
        if not indices_to_do:
            print(f"Taking prepared {self.relative_path} data")
            return

        # print(f"Missing {self.relative_path} data")
        result = False
        while result is False:
            result = mph.run_processes(
                self.generate_data_process, (), self.num_workers,
            )
            if result is False:
                print("Restarting data generation")

    def generate_data_process(self, num_workers, process_id):
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
            exact_normalized_a_torch = Solver.clean_acceleration(
                setting, setting.exact_normalized_a_torch
            )
        else:
            exact_normalized_a_torch = setting.exact_normalized_a_torch

        setting.exact_normalized_a_torch = None
        data = setting.get_data(
            f"{cmh.get_timestamp()} - {index}", exact_normalized_a_torch
        )
        return data

    def check_and_print(
        self, data_count, current_index, setting, step_tqdm, tqdm_description
    ):
        cutoff = config.PRINT_DATA_CUTOFF
        relative_index = current_index % int(data_count * cutoff)
        if relative_index == 0:
            step_tqdm.set_description(
                f"{tqdm_description} - printing data {current_index}"
            )
            plotter_mapper.plot_data_setting(setting, current_index, self.images_path)
        if relative_index == 1:
            step_tqdm.set_description(tqdm_description)

    def __getitem__(self, index):
        return self.get_example(index)

    def __len__(self):
        return self.data_count
