import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from conmech.helpers import cmh, mph, pkh
from conmech.helpers.config import Config
from conmech.scenarios.scenarios import Scenario
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.data.dataset_statistics import DatasetStatistics, FeaturesStatistics
from deep_conmech.graph.scene.scene_input import SceneInput
from deep_conmech.helpers import dch
from deep_conmech.training_config import TrainingConfig


def print_dataset(dataset, cutoff, timestamp, description):
    print(f"Printing dataset {description}...")
    dataloader = get_print_dataloader(dataset)
    batch = next(iter(dataloader))
    iterations = np.min([len(batch), cutoff])


def get_print_dataloader(dataset: "BaseDataset"):
    return get_dataloader(dataset, dataset.config.td.batch_size, num_workers=0, shuffle=False)


def get_valid_dataloader(dataset: "BaseDataset"):
    return get_dataloader(
        dataset,
        dataset.config.td.valid_batch_size,
        num_workers=dataset.config.DATALOADER_WORKERS,
        shuffle=False,
    )


def get_train_dataloader(dataset: "BaseDataset"):
    return get_dataloader(
        dataset,
        dataset.config.td.batch_size,
        num_workers=dataset.config.DATALOADER_WORKERS,
        shuffle=True,
    )


def get_all_dataloader(dataset: "BaseDataset"):
    return get_dataloader(dataset, len(dataset), num_workers=0, shuffle=False)


def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # TODO: #65
    )


def is_memory_overflow(config: TrainingConfig, step_tqdm, tqdm_description):
    memory_usage = dch.get_used_memory_gb()
    step_tqdm.set_description(
        f"{tqdm_description} - memory usage {memory_usage:.2f}/{config.SYNTHETIC_GENERATION_MEMORY_LIMIT_GB}"
    )
    memory_overflow = memory_usage > config.SYNTHETIC_GENERATION_MEMORY_LIMIT_GB
    if memory_overflow:
        step_tqdm.set_description(f"{step_tqdm.desc} - memory overflow")
    return memory_usage > config.SYNTHETIC_GENERATION_MEMORY_LIMIT_GB


def get_process_data_range(process_id, data_part_count):
    return range(process_id * data_part_count, (process_id + 1) * data_part_count)


def get_assigned_scenarios(all_scenarios, num_workers, process_id):
    scenarios_count = len(all_scenarios)
    if scenarios_count % num_workers != 0:
        raise Exception("Cannot divide data generation work")
    assigned_scenarios_count = int(scenarios_count / num_workers)
    assigned_scenarios = all_scenarios[
        process_id * assigned_scenarios_count : (process_id + 1) * assigned_scenarios_count
    ]
    return assigned_scenarios


class BaseDataset:
    def __init__(
        self,
        description: str,
        dimension: int,
        data_count: int,
        randomize_at_load: bool,
        num_workers: int,
        load_to_ram: bool,
        config: TrainingConfig,
    ):
        self.dimension = dimension
        self.description = description
        self.data_count = data_count
        self.randomize_at_load = randomize_at_load
        self.num_workers = num_workers
        self.load_to_ram = load_to_ram
        self.config = config
        self.set_version = 0
        self.all_indices = None
        self.loaded_data = None

    def get_scene_input(self, scenario: Scenario, config: Config) -> SceneInput:
        setting = SceneInput(
            mesh_prop=scenario.mesh_prop,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            config=config,
            create_in_subprocess=False,
        )
        setting.set_randomization(False)
        setting.normalize_and_set_obstacles(scenario.linear_obstacles, scenario.mesh_obstacles)
        return setting

    def get_statistics(self):
        dataloader = get_train_dataloader(self)

        nodes_data = torch.empty((0, SceneInput.nodes_data_dim()))
        edges_data = torch.empty((0, SceneInput.edges_data_dim()))
        for data in cmh.get_tqdm(
            dataloader, config=self.config, desc="Calculating dataset statistics"
        ):
            nodes_data = torch.cat((nodes_data, data.x))
            edges_data = torch.cat((edges_data, data.edge_attr))

        nodes_statistics = FeaturesStatistics(
            nodes_data, SceneInput.get_nodes_data_description(self.dimension)
        )
        edges_statistics = FeaturesStatistics(
            edges_data, SceneInput.get_edges_data_description(self.dimension)
        )

        return DatasetStatistics(
            nodes_statistics=nodes_statistics, edges_statistics=edges_statistics
        )

    def update_data(self):
        pass

    def initialize_data(self):
        cmh.create_folders(self.images_directory)

        self.all_indices = pkh.get_all_indices_pickle(self.data_path)
        if self.data_count == len(self.all_indices):
            settings_path = f"{self.data_path}.settings"
            file_size_gb = os.path.getsize(settings_path) / 1024**3
            print(f"Taking prepared {self.data_id} data ({file_size_gb:.2f} GB)")

        else:
            print("Clearing old data")
            cmh.clear_folder(self.main_directory)
            cmh.create_folders(self.images_directory)

            result = False
            while result is False:
                result = mph.run_processes(
                    self.generate_data_process,
                    (),
                    self.num_workers,
                )
                if result is False:
                    print("Restarting data generation")

            self.all_indices = pkh.get_all_indices_pickle(self.data_path)

        assert self.data_count == len(self.all_indices)
        if self.load_to_ram:
            self.loaded_data = self.load_data_to_ram()
        else:
            self.loaded_data = None
            print("Loading data from disc")

    def load_data_to_ram(self):
        setting_tqdm = cmh.get_tqdm(
            iterable=range(len(self.all_indices)),
            config=self.config,
            desc="Preprocessing and loading dataset to RAM",
        )
        return pkh.get_iterator_pickle(self.data_path, setting_tqdm, self.preprocess_example)

    def generate_data_process(self, num_workers, process_id):
        pass

    @property
    def data_size_id(self):
        pass

    @property
    def data_id(self):
        td = self.config.td
        return f"{self.description}_m:{td.mesh_density}_{self.data_size_id}"

    @property
    def main_directory(self):
        return f"./{self.config.datasets_main_path}/{self.data_id}"

    @property
    def data_path(self):
        return f"{self.main_directory}/DATA_{self.set_version}"

    @property
    def images_directory(self):
        return f"{self.main_directory}/images_{self.set_version}"

    def get_example(self, index):
        if self.loaded_data is not None:
            return self.loaded_data[index]
        else:
            with pkh.open_file_settings_read_pickle(self.data_path) as file:
                setting = pkh.load_index_pickle(
                    index=index, all_indices=self.all_indices, settings_file=file
                )
        data = self.preprocess_example(setting, index)
        return data

    def preprocess_example(self, setting, index):
        if self.randomize_at_load:
            setting.set_randomization(True)
            exact_normalized_a_torch = Calculator.clean_acceleration(
                setting, setting.exact_normalized_a_torch
            )
        else:
            exact_normalized_a_torch = setting.exact_normalized_a_torch

        setting.exact_normalized_a_torch = None
        data = setting.get_data(
            f"{cmh.get_timestamp(self.config)} - {index}", exact_normalized_a_torch
        )
        return data

    def check_and_print(self, data_count, current_index, setting, step_tqdm, tqdm_description):
        plot_index_skip = int(data_count * (1 / self.config.dataset_images_count))
        relative_index = 1 if plot_index_skip == 0 else current_index % plot_index_skip
        if relative_index == 0:
            step_tqdm.set_description(f"{tqdm_description} - plotting index {current_index}")
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
