import os
import pickle
from typing import Iterable, Optional

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
    _ = timestamp
    print(f"Printing dataset {description}...")
    dataloader = get_print_dataloader(dataset)
    batch = next(iter(dataloader))
    iterations = np.min([len(batch), cutoff])
    _ = iterations


def get_print_dataloader(dataset: "BaseDataset"):
    return get_dataloader(dataset, dataset.config.td.batch_size, num_workers=0, shuffle=False)


def get_valid_dataloader(dataset: "BaseDataset"):
    return get_dataloader(
        dataset,
        dataset.config.td.valid_batch_size,
        num_workers=dataset.config.dataloader_workers,
        shuffle=False,
    )


def get_train_dataloader(dataset: "BaseDataset"):
    return get_dataloader(
        dataset,
        dataset.config.td.batch_size,
        num_workers=dataset.config.dataloader_workers,
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
        pin_memory=False,  # True,  # TODO: #65
    )


def is_memory_overflow(config: TrainingConfig, step_tqdm, tqdm_description):
    memory_usage = dch.get_used_memory_gb()
    step_tqdm.set_description(
        f"{tqdm_description} - memory usage {memory_usage:.2f}/{config.synthetic_generation_memory_limit_gb}"
    )
    memory_overflow = memory_usage > config.synthetic_generation_memory_limit_gb
    if memory_overflow:
        step_tqdm.set_description(f"{step_tqdm.desc} - memory overflow")
    return memory_usage > config.synthetic_generation_memory_limit_gb


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


def get_scene_input(scenario: Scenario, config: Config) -> SceneInput:
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


class BaseDataset:
    def __init__(
        self,
        description: str,
        dimension: int,
        data_count: int,
        randomize_at_load: bool,
        num_workers: int,
        load_features_to_ram: bool,
        load_targets_to_ram: bool,
        with_scenes_file: bool,
        config: TrainingConfig,
    ):
        self.dimension = dimension
        self.description = description
        self.data_count = data_count
        self.randomize_at_load = randomize_at_load
        self.num_workers = num_workers
        self.load_features_to_ram = load_features_to_ram
        self.load_targets_to_ram = load_targets_to_ram
        self.with_scenes_file = with_scenes_file
        self.config = config
        self.scene_indices = None
        self.loaded_features_data = None
        self.loaded_targets_data = None
        self.features_indices = None
        self.targets_indices = None

    @property
    def data_size_id(self):
        pass

    @property
    def data_id(self):
        td = self.config.td
        return f"{self.description}_d={td.dimension}_m={td.mesh_density}_{self.data_size_id}"

    @property
    def main_directory(self):
        return f"./{self.config.datasets_main_path}/{self.data_id}"

    @property
    def images_directory(self):
        return f"{self.main_directory}/images"

    @property
    def tmp_directory(self):
        return f"{self.main_directory}/tmp"

    @property
    def data_path(self):
        return f"{self.main_directory}/DATA"

    @property
    def scenes_data_path(self):
        return f"{self.main_directory}/DATA.scenes"

    @property
    def features_data_path(self):
        return f"{self.tmp_directory}/DATASET.feat"

    @property
    def targets_data_path(self):
        return f"{self.tmp_directory}/DATASET.targ"

    def initialize_data(self):
        print(f"----INITIALIZING DATASET ({self.data_id})----")
        self.create_folders()
        if self.with_scenes_file:
            self.initialize_scenes()
        else:
            print("Skipping scenes file generation")
        self.initialize_features_and_targets()

        if self.load_features_to_ram:
            self.load_features()
        else:
            print("Reading features from disc")

        if self.load_targets_to_ram:
            self.load_targets()
        else:
            print("Reading targets from disc")

    def create_folders(self):
        cmh.create_folders(self.images_directory)
        cmh.create_folders(self.tmp_directory)

    def initialize_scenes(self):
        self.scene_indices = pkh.get_all_indices(self.scenes_data_path)
        if self.data_count == len(self.scene_indices):
            file_size_gb = os.path.getsize(self.scenes_data_path) / 1024**3
            print(f"Taking prepared scenes ({file_size_gb:.2f} GB)")

        else:
            print("Clearing old data")
            cmh.clear_folder(self.main_directory)
            self.create_folders()

            result = False
            while result is False:
                result = mph.run_processes(
                    self.generate_data_process,
                    (),
                    self.num_workers,
                )
                if result is False:
                    print("Restarting data generation")

            self.scene_indices = pkh.get_all_indices(self.scenes_data_path)
        assert self.data_count == len(self.scene_indices)

    def get_iterator(self, data_tqdm: Iterable[int], scenes_path: Optional[str] = None):
        for index in data_tqdm:
            if scenes_path is not None:
                with open(scenes_path, "rb") as file:
                    scene = pickle.load(file)
            else:
                scene, _ = self.generate_scene(index)

            features_data, target_data = self.preprocess_example(scene, index)
            yield (features_data, target_data)

    def initialize_features_and_targets(self):
        self.features_indices = pkh.get_all_indices(self.features_data_path)
        self.targets_indices = pkh.get_all_indices(self.targets_data_path)
        if self.data_count == len(self.features_indices) and self.data_count == len(
            self.targets_indices
        ):
            features_size_gb = os.path.getsize(self.features_data_path) / 1024**3
            targets_size_gb = os.path.getsize(self.targets_data_path) / 1024**3
            print(
                f"Taking prepared features ({features_size_gb:.2f} GB) and targets ({targets_size_gb:.2f} GB) dataset"
            )
        else:
            data_tqdm = cmh.get_tqdm(
                iterable=range(self.data_count),
                config=self.config,
                desc="Preprocessing dataset",
            )

            features_file, features_indices_file = pkh.open_files_append(self.features_data_path)
            targets_file, targets_indices_file = pkh.open_files_append(self.targets_data_path)

            scenes_path = self.scenes_data_path if self.with_scenes_file else None
            with features_file, features_indices_file, targets_file, targets_indices_file:
                for features_data, targets_data in self.get_iterator(
                    data_tqdm=data_tqdm, scenes_path=scenes_path
                ):
                    pkh.append_data(features_data, features_file, features_indices_file)
                    pkh.append_data(targets_data, targets_file, targets_indices_file)

            self.features_indices = pkh.get_all_indices(self.features_data_path)
            self.targets_indices = pkh.get_all_indices(self.targets_data_path)

        assert self.data_count == len(self.features_indices)
        assert self.data_count == len(self.targets_indices)

    def load_features(self):
        self.loaded_features_data = self.get_data_loaded_to_ram(
            "features", self.features_data_path, self.features_indices
        )

    def load_targets(self):
        self.loaded_targets_data = self.get_data_loaded_to_ram(
            "targets", self.targets_data_path, self.targets_indices
        )

    def get_data_loaded_to_ram(self, desc, data_path, indices):
        data_tqdm = cmh.get_tqdm(
            iterable=range(len(indices)),
            config=self.config,
            desc=f"Loading {desc} to RAM",
        )
        file = pkh.open_file_read(data_path)
        with file:
            data = [pkh.load_index(index, indices, file) for index in data_tqdm]
        return data

    def get_statistics(self):
        dataloader = get_train_dataloader(self)

        dimension = self.config.td.dimension
        nodes_data = torch.empty((0, SceneInput.nodes_data_dim(dimension)))
        edges_data = torch.empty((0, SceneInput.edges_data_dim(dimension)))
        for features_data in cmh.get_tqdm(
            dataloader, config=self.config, desc="Calculating dataset statistics"
        ):
            nodes_data = torch.cat((nodes_data, features_data.x))
            edges_data = torch.cat((edges_data, features_data.edge_attr))

        nodes_statistics = FeaturesStatistics(
            nodes_data, SceneInput.get_nodes_data_description(self.dimension)
        )
        edges_statistics = FeaturesStatistics(
            edges_data, SceneInput.get_edges_data_description(self.dimension)
        )

        return DatasetStatistics(
            nodes_statistics=nodes_statistics, edges_statistics=edges_statistics
        )

    def get_features_data(self, index):
        if self.loaded_features_data is not None:
            return self.loaded_features_data[index]
        with pkh.open_file_read(self.features_data_path) as file:
            features_data = pkh.load_index(
                index=index, all_indices=self.features_indices, data_file=file
            )
        return features_data

    def get_targets_data(self, index):
        if self.loaded_targets_data is not None:
            return self.loaded_targets_data[index]
        with pkh.open_file_read(self.targets_data_path) as file:
            target_data = pkh.load_index(
                index=index, all_indices=self.targets_indices, data_file=file
            )
        return target_data

    def preprocess_example(self, scene, index):
        if self.randomize_at_load:
            scene.set_randomization(True)
            exact_normalized_a_torch = Calculator.clean_acceleration(
                scene, scene.exact_normalized_a_torch
            )
        else:
            exact_normalized_a_torch = scene.exact_normalized_a_torch

        scene.exact_normalized_a_torch = None
        features_data, target_data = scene.get_data(index, exact_normalized_a_torch)
        return features_data, target_data

    def check_and_print(self, data_count, current_index, scene, step_tqdm, tqdm_description):
        plot_index_skip = int(data_count * (1 / self.config.dataset_images_count))
        relative_index = 1 if plot_index_skip == 0 else current_index % plot_index_skip
        if relative_index == 0:
            step_tqdm.set_description(f"{tqdm_description} - plotting index {current_index}")
            self.plot_data_setting(scene, current_index, self.images_directory)
        if relative_index == 1:
            step_tqdm.set_description(tqdm_description)

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

    def generate_data_process(self, num_workers: int, process_id: int):
        pass

    def generate_scene(self, index: int):
        _ = index
        return None, None

    def __getitem__(self, index):
        return self.get_features_data(index)

    def __len__(self):
        return self.data_count
