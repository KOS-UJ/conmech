import copy
import multiprocessing
import os
from ctypes import ArgumentError
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from conmech.helpers import cmh, mph, pkh
from conmech.scene.scene import Scene
from conmech.simulations import simulation_runner
from deep_conmech.data.dataset_statistics import DatasetStatistics, FeaturesStatistics
from deep_conmech.scene.scene_input import GraphData, SceneInput
from deep_conmech.training_config import TrainingConfig


def print_dataset(dataset, cutoff, timestamp, description):
    _ = timestamp
    print(f"Printing dataset {description}...")
    dataloader = get_print_dataloader(dataset)
    batch = next(iter(dataloader))
    iterations = np.min([len(batch), cutoff])
    _ = iterations


def get_print_dataloader(dataset: "BaseDataset", rank: int, world_size: int):
    return get_dataloader(
        dataset=dataset,
        rank=rank,
        world_size=world_size,
        batch_size=dataset.config.td.batch_size,
        num_workers=0,
        shuffle=False,
    )


def get_valid_dataloader(dataset: "BaseDataset", rank: int, world_size: int):
    return get_dataloader(
        dataset=dataset,
        rank=rank,
        world_size=world_size,
        batch_size=dataset.config.td.valid_batch_size,
        num_workers=dataset.config.dataloader_workers,
        shuffle=False,
    )


def get_train_dataloader(dataset: "BaseDataset", rank: int, world_size: int):
    return get_dataloader(
        dataset=dataset,
        rank=rank,
        world_size=world_size,
        batch_size=dataset.config.td.batch_size,
        num_workers=dataset.config.dataloader_workers,
        shuffle=True,
    )


def get_all_dataloader(dataset: "BaseDataset", rank: int, world_size: int):
    return get_dataloader(dataset, world_size, rank, len(dataset), num_workers=0, shuffle=False)


def get_dataloader(
    dataset, rank: int, world_size: int, batch_size: int, num_workers: int, shuffle: bool
):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,  # TODO: #65
        persistent_workers=False,  # True,
        # prefetch_factor=10,
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
        with_scenes_file: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
    ):
        self.dimension = dimension
        self.description = description
        self.data_count = data_count
        self.layers_count = layers_count
        self.randomize_at_load = randomize_at_load
        self.num_workers = num_workers
        self.load_features_to_ram = load_features_to_ram
        self.with_scenes_file = with_scenes_file
        self.config = config
        self.scene_indices = None
        self.loaded_features_data = None
        self.features_indices = None
        self.files_lock = mph.get_lock()
        self.rank = rank
        self.world_size = world_size
        self.file = None

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
            print(f"Taking prepared dataset ({self.get_size(self.features_data_path):.2f} GB)")
            return

        if self.with_scenes_file:
            self.initialize_scenes()
        else:
            print("Skipping scenes file generation")

        # cmh.profile(self.initialize_features_and_targets_process)
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

        # mph.run_process(self.generate_data_process)
        done = mph.run_processes(self.generate_data_process, num_workers=self.num_workers)
        if not done:
            print("NOT DONE")

        self.scene_indices = self.get_all_scene_indices()
        assert self.data_count == len(self.scene_indices)

    def get_scene_from_file(self, scene_index: int):
        with pkh.open_file_read(self.scenes_data_path) as data_file:
            scene = pkh.load_byte_index(
                byte_index=self.scene_indices[scene_index],
                data_file=data_file,
            )
        return scene

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
        return self.data_count == len(self.features_indices)

    def load_indices(self):
        self.features_indices = pkh.get_all_indices(self.features_data_path)[: self.data_count]

    def get_size(self, data_path):
        return os.path.getsize(data_path) / 1024**3

    def initialize_features_and_targets_process(self, num_workers: int = 1, process_id: int = 0):
        assigned_data_range = self.get_process_data_range(
            data_count=self.data_count, process_id=process_id, num_workers=num_workers
        )
        data_tqdm = cmh.get_tqdm(
            iterable=assigned_data_range,
            config=self.config,
            desc=f"Preprocessing data - process {process_id+1}/{num_workers}",
            position=process_id,
        )
        for scene in self.get_scenes_iterator(data_tqdm=data_tqdm):
            layers_list = [
                scene.get_features_data(
                    layer_number=layer_number,
                )
                for layer_number in range(self.layers_count)
            ]
            target_data = scene.get_target_data()
            graph_data = GraphData(layer_list=layers_list, target_data=target_data)
            pkh.append_data(
                data=graph_data,
                data_path=self.features_data_path,
                lock=self.files_lock,
            )
        return True

    def load_features(self):

        # shared_array_base = multiprocessing.Array(type(GraphData), 10)
        # shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        # self.shared_array = torch.from_numpy(shared_array)
        manager = multiprocessing.Manager()
        self.loaded_features_data = []  # manager.dict()

        iterable = self.get_process_data_range(
            len(self.features_indices), process_id=self.rank, num_workers=self.world_size
        )
        data_tqdm = cmh.get_tqdm(
            iterable=iterable,
            config=self.config,
            desc=f"Loading data to RAM",
            position=self.rank,
        )
        i = 0
        with pkh.open_file_read(self.features_data_path) as file:
            for index in data_tqdm:
                if self.is_loaded_data_memory_overflow:
                    raise ArgumentError
                example = pkh.load_byte_index(self.features_indices[index], file)
                self.loaded_features_data.append(example)
                # self.loaded_features_data[i] = example
                # i += 1

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

    def get_features_and_targets_data(self, index) -> GraphData:
        if self.loaded_features_data is not None:
            return self.loaded_features_data[index]
        # shifted_index = index + self.rank * self.data_count // self.world_size
        shifted_index = index
        with pkh.open_file_read(self.features_data_path) as file:
            features_data = pkh.load_byte_index(
                byte_index=self.features_indices[shifted_index], data_file=file  # self.file
            )
        return features_data

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

    def open_file(self):
        if self.file is None:
            self.file = pkh.open_file_read(self.features_data_path)

    def __getitem__(self, index: int):
        # self.open_file()
        graph_data = self.get_features_and_targets_data(index)
        return graph_data.layer_list, graph_data.target_data

    def __len__(self):
        return self.data_count  # // self.world_size
