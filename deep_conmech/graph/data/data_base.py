import re
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from conmech.helpers import cmh, mph
from conmech.helpers.config import Config
from deep_conmech.common import simulation_runner
from deep_conmech.common.training_config import TrainingConfig
from deep_conmech.graph.data.dataset_statistics import (DatasetStatistics,
                                                        FeaturesStatistics)
from deep_conmech.graph.helpers import dch, thh
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_iterable import SettingIterable
from deep_conmech.simulator.solver import Solver
from torch_geometric.loader import DataLoader


def print_dataset(dataset, cutoff, timestamp, description):
    print(f"Printing dataset {description}...")
    dataloader = get_print_dataloader(dataset)
    batch = next(iter(dataloader))
    iterations = np.min([len(batch), cutoff])



def get_print_dataloader(dataset):
    return get_dataloader(
        dataset, dataset.config.BATCH_SIZE, num_workers=0, shuffle=False
    )


def get_valid_dataloader(dataset):
    return get_dataloader(
        dataset,
        dataset.config.td.VALID_BATCH_SIZE,
        num_workers=dataset.config.DATALOADER_WORKERS,
        shuffle=False,
    )


def get_train_dataloader(dataset):
    return get_dataloader(
        dataset,
        dataset.config.td.BATCH_SIZE,
        num_workers=dataset.config.DATALOADER_WORKERS,
        shuffle=True,
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


def is_memory_overflow(config: TrainingConfig, step_tqdm, tqdm_description):
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



def get_assigned_scenarios(all_scenarios, num_workers, process_id):
    scenarios_count = len(all_scenarios)
    if scenarios_count % num_workers != 0:
        raise Exception("Cannot divide data generation work")
    assigned_scenarios_count = int(scenarios_count / num_workers)
    assigned_scenarios = all_scenarios[
                         process_id
                         * assigned_scenarios_count: (process_id + 1)
                                                     * assigned_scenarios_count
                         ]
    return assigned_scenarios


class BaseDatasetDynamic:
    def __init__(
        self,
        dimension,
        relative_path,
        data_count,
        randomize_at_load,
        num_workers,
        config: TrainingConfig,
    ):
        self.dimension = dimension
        self.relative_path = relative_path
        self.data_count = data_count
        self.randomize_at_load = randomize_at_load
        self.num_workers = num_workers
        self.config = config

    def get_setting_input(self, scenario: Scenario, config: Config) -> SettingInput:
        setting = SettingInput(
            mesh_data=scenario.mesh_data,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            config=config,
            create_in_subprocess=False,
        )
        setting.set_randomization(False)
        setting.set_obstacles(scenario.obstacles)
        return setting

    def get_statistics(self):
        dataloader = get_train_dataloader(self)

        nodes_data = torch.empty((0, SettingInput.nodes_data_dim()))
        edges_data = torch.empty((0, SettingInput.edges_data_dim()))
        for data in cmh.get_tqdm(dataloader, config=self.config,
                                 desc="Calculating dataset statistics"):
            nodes_data = torch.cat((nodes_data, data.x))
            edges_data = torch.cat((edges_data, data.edge_attr))

        nodes_statistics = FeaturesStatistics(
            nodes_data, SettingInput.get_nodes_data_description(self.dimension)
        )
        edges_statistics = FeaturesStatistics(
            edges_data, SettingInput.get_edges_data_description(self.dimension)
        )

        return DatasetStatistics(
            nodes_statistics=nodes_statistics, edges_statistics=edges_statistics
        )

    def update_data(self):
        pass

    def clear_and_initialize_data(self):
        print(f"Clearing {self.relative_path} data")
        cmh.clear_folder(self.main_directory)
        cmh.clear_folder(self.images_directory)
        self.initialize_data()

    def initialize_data(self):
        cmh.create_folders(self.main_directory)
        cmh.create_folders(self.images_directory)

        self.all_indices = SettingIterable.get_all_indices_pickle(self.data_path)
        if self.data_count == len(self.all_indices):
            print(f"Taking prepared {self.relative_path} data")
        else:
            result = False
            while result is False:
                result = mph.run_processes(
                    self.generate_data_process, (), self.num_workers,
                )
                if result is False:
                    print("Restarting data generation")
        
            self.all_indices = SettingIterable.get_all_indices_pickle(self.data_path)
        
        self.settings_file = SettingIterable.open_file_settings_read_pickle(self.data_path) # TODO: Close after training

    def generate_data_process(self, num_workers, process_id):
        pass

    @property
    def main_directory(self):
        return f"./datasets/{self.config.DATA_FOLDER}/{self.relative_path}"

    @property
    def data_path(self):
        return f"{self.main_directory}/DATA"

    @property
    def images_directory(self):
        return f"{self.main_directory}/images"


    def get_example(self, index):
        setting = SettingIterable.load_index_pickle(index=index, all_indices=self.all_indices, settings_file=self.settings_file)
        if self.randomize_at_load:
            setting.set_randomization(True)
            exact_normalized_a_torch = Solver.clean_acceleration(
                setting, setting.exact_normalized_a_torch
            )
        else:
            exact_normalized_a_torch = setting.exact_normalized_a_torch

        setting.exact_normalized_a_torch = None
        data = setting.get_data(
            f"{cmh.get_timestamp(self.config)} - {index}", exact_normalized_a_torch
        )
        return data

    def check_and_print(
            self, data_count, current_index, setting, step_tqdm, tqdm_description
    ):
        cutoff = self.config.PRINT_DATA_CUTOFF
        relative_index = current_index % int(data_count * cutoff)
        if relative_index == 0:
            step_tqdm.set_description(
                f"{tqdm_description} - printing data {current_index}"
            )
            self.plot_data_setting(setting, current_index, self.images_directory)
        if relative_index == 1:
            step_tqdm.set_description(tqdm_description)

    def __getitem__(self, index):
        return self.get_example(index)

    def __len__(self):
        return self.data_count

    def plot_data_setting(self, setting, filename, catalog):
        cmh.create_folders(catalog)
        extension = "png"  # pdf
        path = f"{catalog}/{filename}.{extension}"
        simulation_runner.plot_setting(
            current_time=0,
            setting=setting,
            path=path,
            base_setting=None,
            draw_detailed=True,
            extension=extension,
        )
