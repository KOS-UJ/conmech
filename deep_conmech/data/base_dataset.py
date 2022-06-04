import copy
import os
from ctypes import ArgumentError
from typing import Iterable

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from conmech.helpers import cmh, mph, pkh
from conmech.scene.scene import Scene
from conmech.simulations import simulation_runner
from deep_conmech.data.dataset_statistics import DatasetStatistics, FeaturesStatistics
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_input import SceneInput, TargetData
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
        pin_memory=True,  # True,  # TODO: #65
    )


class BaseDataset:
    def __init__(
        self,
        description: str,
        dimension: int,
        data_count: int,
        layers_count: int,
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
        self.layers_count = layers_count
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
        self.device = thh.device(self.config)
        self.files_lock = mph.get_lock()

    @property
    def data_size_id(self):
        pass

    @property
    def is_synthetic_generation_memory_overflow(self):
        memory_usage = cmh.get_used_memory_gb()
        return memory_usage > self.config.synthetic_generation_memory_limit_gb

    @property
    def is_loaded_data_memory_overflow(self):
        memory_usage = cmh.get_used_memory_gb()
        return memory_usage > self.config.loaded_data_memory_limit_gb

    @property
    def data_id(self):
        td = self.config.td
        return f"{self.description}_d:{td.dimension}_m:{td.mesh_density}_{self.data_size_id}"

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

    def get_process_data_range(self, data_count: int, process_id: int, num_workers: int):
        scenes_part_count = int(data_count / num_workers)
        if process_id == num_workers - 1:
            return range(process_id * scenes_part_count, data_count)
        return range(process_id * scenes_part_count, (process_id + 1) * scenes_part_count)

    def initialize_data(self):
        print(f"----INITIALIZING DATASET ({self.data_id})----")
        self.create_folders()
        self.load_indices()
        if self.check_indices():
            print(
                f"Taking prepared features ({self.get_size(self.features_data_path):.2f} GB) and targets ({self.get_size(self.targets_data_path):.2f} GB) dataset"
            )
            return

        if self.with_scenes_file:
            self.initialize_scenes()
        else:
            print("Skipping scenes file generation")

        mph.run_processes(
            self.initialize_features_and_targets_process, num_workers=self.num_workers
        )
        self.load_indices()
        assert self.check_indices()

    def load_data(self):
        print(f"----LOADING DATASET ({self.data_id})----")
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

    def get_all_scene_indices(self):
        return pkh.get_all_indices(self.scenes_data_path)[: self.data_count]

    def initialize_scenes(self):
        # cmh.profile(self.generate_data_process)

        self.scene_indices = self.get_all_scene_indices()
        if self.data_count == len(self.scene_indices):
            print(f"Taking prepared scenes ({self.get_size(self.scenes_data_path):.2f} GB)")
            return

        print("Clearing old data")
        cmh.clear_folder(self.main_directory)
        self.create_folders()

        done = mph.run_processes(self.generate_data_process, num_workers=self.num_workers)
        if not done:
            print("NOT DONE")

        # mph.run_process(self.generate_data_process)

        self.scene_indices = self.get_all_scene_indices()
        assert self.data_count == len(self.scene_indices)

    def get_scene_from_file(self, scene_index: int):
        return pkh.load_index(
            index=scene_index,
            all_indices=self.scene_indices,
            data_file=pkh.open_file_read(self.scenes_data_path),
        )

    def get_scenes_iterator(self, data_tqdm: Iterable[int]):
        for scene_index in data_tqdm:
            if self.with_scenes_file:
                scene = self.get_scene_from_file(scene_index)
            else:
                scene = self.generate_scene()

            if self.randomize_at_load:
                scene.set_randomization(self.config)

            # exact_normalized_a_torch = Calculator.clean_acceleration(
            #    scene_input, scene_input.exact_normalized_a_torch)
            # else:
            #   exact_normalized_a_torch = scene_input.exact_normalized_a_torch

            yield scene

    def check_indices(self):
        return self.data_count == len(self.features_indices) and self.data_count == len(
            self.targets_indices
        )

    def load_indices(self):
        self.features_indices = pkh.get_all_indices(self.features_data_path)[: self.data_count]
        self.targets_indices = pkh.get_all_indices(self.targets_data_path)[: self.data_count]

    def get_size(self, data_path):
        return os.path.getsize(data_path) / 1024**3

    def initialize_features_and_targets_process(self, num_workers: int, process_id: int):
        assigned_data_range = self.get_process_data_range(
            data_count=self.data_count, process_id=process_id, num_workers=num_workers
        )
        data_tqdm = cmh.get_tqdm(
            iterable=assigned_data_range,
            config=self.config,
            desc=f"Preprocessing dataset - process {process_id+1}/{num_workers}",
            position=process_id,
        )
        for scene in self.get_scenes_iterator(data_tqdm=data_tqdm):
            target_data = scene.get_target_data()
            features_layers_list = [
                scene.get_features_data(
                    layer_number=layer_number,
                )
                for layer_number in range(self.layers_count)
            ]
            pkh.append_multiple_data(
                all_data=[target_data, features_layers_list],
                all_data_paths=[self.targets_data_path, self.features_data_path],
                lock=self.files_lock,
            )
        return True

    def load_features(self):
        self.loaded_features_data = self.get_data_loaded_to_ram(
            "features", self.features_data_path, self.features_indices
        )
        # self.loaded_features_data = []
        # for layer_list in data:
        #     self.loaded_features_data.append([layer.to(self.device) for layer in layer_list])
        # a = 0

    def load_targets(self):
        self.loaded_targets_data = self.get_data_loaded_to_ram(
            "targets", self.targets_data_path, self.targets_indices
        )
        # self.loaded_targets_data = [
        #     TargetData(a_correction=example["a_correction"], energy_args=example["args"]).to(
        #         self.device
        #     )
        #     for example in data
        # ]

    def get_data_loaded_to_ram(self, desc, data_path, indices):
        data_tqdm = cmh.get_tqdm(
            iterable=range(len(indices)),
            config=self.config,
            desc=f"Loading {desc} to RAM",
        )
        file = pkh.open_file_read(data_path)
        data = []
        with file:
            for index in data_tqdm:
                if self.is_loaded_data_memory_overflow:
                    raise ArgumentError
                example = pkh.load_index(index, indices, file)
                data.append(example)
        return data

    def get_statistics(self, layer_number):
        dataloader = get_train_dataloader(self)

        dimension = self.config.td.dimension
        nodes_data = torch.empty((0, SceneInput.get_nodes_data_dim(dimension)))
        edges_data = torch.empty((0, SceneInput.get_edges_data_dim(dimension)))
        for layers_list in cmh.get_tqdm(
            dataloader, config=self.config, desc="Calculating dataset statistics"
        ):
            layer = layers_list[layer_number]
            nodes_data = torch.cat((nodes_data, layer.x))
            edges_data = torch.cat((edges_data, layer.edge_attr))

        nodes_statistics = FeaturesStatistics(
            nodes_data, SceneInput.get_nodes_data_description(self.dimension)
        )
        edges_statistics = FeaturesStatistics(
            edges_data, SceneInput.get_edges_data_description(self.dimension)
        )

        return DatasetStatistics(
            nodes_statistics=nodes_statistics, edges_statistics=edges_statistics
        )

    def get_features_data(self, scene_index):
        if self.loaded_features_data is not None:
            return self.loaded_features_data[scene_index]
        with pkh.open_file_read(self.features_data_path) as file:
            features_data = pkh.load_index(
                index=scene_index, all_indices=self.features_indices, data_file=file
            )
        return features_data

    def get_targets_data(self, index: int):
        if self.loaded_targets_data is not None:
            target_data = self.loaded_targets_data[index]
        else:
            with pkh.open_file_read(self.targets_data_path) as file:
                target_data = pkh.load_index(
                    index=index, all_indices=self.targets_indices, data_file=file
                )
        return TargetData(a_correction=target_data["a_correction"], energy_args=target_data["args"])

    def check_and_print(self, all_data_count, current_index, scene, step_tqdm, tqdm_description):
        images_count = self.config.dataset_images_count
        if images_count is None:
            return
        plot_index_skip = 1 if all_data_count < images_count else int(all_data_count / images_count)
        relative_index = 1 if plot_index_skip == 0 else current_index % plot_index_skip
        if relative_index == 0:
            step_tqdm.set_description(f"{tqdm_description} - plotting index {current_index}")
            self.plot_data_scene(scene, current_index, self.images_directory)
        if relative_index == 1:
            step_tqdm.set_description(tqdm_description)

    def plot_data_scene(self, scene: Scene, filename, catalog):
        cmh.create_folders(catalog)
        extension = "png"  # pdf
        path = f"{catalog}/{filename}.{extension}"
        simulation_runner.plot_setting(
            current_time=0,
            scene=scene,
            path=path,
            base_scene=None,
            draw_detailed=True,
            extension=extension,
        )

    def generate_data_process(self, num_workers: int = 1, process_id: int = 0):
        pass

    def safe_save_scene(self, scene, data_path: str):
        scene_copy = copy.deepcopy(scene)
        scene_copy.prepare_to_save()
        pkh.append_data(
            data=scene_copy,
            data_path=data_path,
            lock=self.files_lock,
        )

    def __getitem__(self, index):
        layers_list = self.get_features_data(index)[: self.layers_count]
        layers_list[0].scene_id = index
        return layers_list

    def __len__(self):
        return self.data_count
