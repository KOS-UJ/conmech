import copy
import os
from ctypes import ArgumentError
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from conmech.helpers import cmh, mph, pkh
from conmech.plotting.plotter_functions import plot_setting
from deep_conmech.data.data_classes import GraphData
from deep_conmech.data.dataset_statistics import (
    FeaturesStatistics,
    FeaturesStatisticsPandas,
)
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
        load_data=False,
    )


def get_valid_dataloader(dataset: "BaseDataset"):
    return get_dataloader(
        dataset=dataset,
        rank=dataset.rank,
        world_size=dataset.world_size,
        batch_size=dataset.config.td.batch_size,
        num_workers=dataset.config.dataloader_workers,
        shuffle=False,
        load_data=True,  # False,
    )


def get_train_dataloader(dataset: "BaseDataset"):
    return get_dataloader(
        dataset=dataset,
        rank=dataset.rank,
        world_size=dataset.world_size,
        batch_size=dataset.config.td.batch_size,
        num_workers=dataset.config.dataloader_workers,
        shuffle=True,
        load_data=True,
    )


def get_dataloader(
    dataset,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    load_data: bool,
    collate_fn=None,
):

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if load_data else None,
        # prefetch_factor=10,
        collate_fn=collate_fn,
    )


def worker_init_fn(worker_id: int):
    _ = worker_id
    get_worker_info().dataset.load_data()


class BaseDataset:
    def __init__(
        self,
        description: str,
        use_jax: bool,
        dimension: int,
        data_count: int,
        solve_function: Callable,
        load_data_to_ram: bool,
        randomize: bool,
        num_workers: int,
        with_scenes_file: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
        device_count: int,
        item_fn: Callable = None,
    ):
        self.dimension = dimension
        self.use_jax = use_jax
        self.description = description
        self.data_count = data_count
        self.solve_function = solve_function
        self.load_data_to_ram = load_data_to_ram
        self.randomize = randomize
        self.num_workers = num_workers
        self.with_scenes_file = with_scenes_file
        self.config = config
        self.scene_indices = None
        self.features_indices = None
        self.files_lock = None if world_size < 2 else mph.get_lock()
        self.rank = rank
        self.world_size = world_size
        self.file = None
        self.loaded_data = None
        self.data_file = None
        self.device_count = device_count
        self.item_fn = item_fn
        self.statistics = None

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

    def unload_and_clear_indices(self):
        self.features_indices = None
        cmh.clear_folder(self.tmp_directory)
        cmh.create_folder(self.tmp_directory)

    def initialize_data(self):
        print(f"----NODE {self.rank}: INITIALIZING DATASET ({self.data_id})----")
        self.create_folders()
        self.load_indices()

        if self.check_indices():
            print(f"Taking prepared dataset ({self.get_size(self.features_data_path):.2f} GB)")
            return

        self.unload_and_clear_indices()
        if not self.with_scenes_file:
            print("Skipping scenes file generation")

        self.initialize_scenes()

        self.load_indices()
        assert self.check_indices()

    def create_folders(self):
        cmh.create_folders(self.images_directory)
        cmh.create_folders(self.tmp_directory)

    def get_all_scene_indices(self):
        return pkh.get_all_indices(self.scenes_data_path)[: self.data_count]

    def generate_data(self):
        pass

    def initialize_scenes(self):
        # cmh.profile(self.generate_data_process)

        self.scene_indices = self.get_all_scene_indices()
        if self.data_count == len(self.scene_indices):
            print(f"Taking prepared scenes ({self.get_size(self.scenes_data_path):.2f} GB)")
            return

        print("Clearing old data")
        cmh.clear_folder(self.main_directory)
        self.create_folders()

        self.generate_data()

        if self.with_scenes_file:
            self.scene_indices = self.get_all_scene_indices()
            assert self.data_count == len(self.scene_indices)
            cmh.profile(self.initialize_features_and_targets_process, baypass=True)
            # mph.run_processes(
            #     self.initialize_features_and_targets_process, num_workers=self.num_workers
            # )

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

    def save_features_and_target(self, scene):
        layers_list = [
            scene.get_features_data(layer_number=layer_number, to_cpu=True)
            for layer_number in range(len(scene.all_layers))
        ]
        target_data = scene.get_target_data(to_cpu=True)

        graph_data = GraphData(layer_list=layers_list, target_data=target_data, scene=None)
        pkh.append_data(
            data=graph_data,
            data_path=self.features_data_path,
            lock=self.files_lock,
        )

    def initialize_features_and_targets_process(self, num_workers: int = 1, process_id: int = 0):
        assigned_data_range = range(process_id, self.data_count, num_workers)
        data_tqdm = cmh.get_tqdm(
            iterable=assigned_data_range,
            config=self.config,
            desc=f"Preprocessing data - process {process_id+1}/{num_workers}",
            position=process_id,
        )
        for scene in self.get_scenes_iterator(data_tqdm=data_tqdm):
            self.save_features_and_target(scene)
        return True

    def get_statistics(self):
        saved_device_count = self.device_count
        self.device_count = 1

        dataloader = get_dataloader(
            dataset=self,
            rank=0,
            world_size=1,
            batch_size=32,  # 32,  # 1,  # self.config.td.batch_size,
            num_workers=0,
            shuffle=True,
            load_data=True,
        )

        statistics = {
            "sparse_nodes": FeaturesStatistics(),
            "sparse_edges": FeaturesStatistics(),
            "multilayer_edges": FeaturesStatistics(),
            "dense_nodes": FeaturesStatistics(),
            "dense_edges": FeaturesStatistics(),
            "target_normalized_new_displacement": FeaturesStatistics(),
        }

        for graph_data in cmh.get_tqdm(
            dataloader, config=self.config, desc="Calculating dataset mean and max abs"
        ):
            assert len(graph_data) == 1
            (dense_layer, sparse_layer), target_layer = graph_data[0]
            statistics["sparse_nodes"].set_mean_and_max_abs(sparse_layer.x)
            statistics["sparse_edges"].set_mean_and_max_abs(sparse_layer.edge_attr)
            statistics["multilayer_edges"].set_mean_and_max_abs(sparse_layer.edge_attr_to_down)
            statistics["dense_nodes"].set_mean_and_max_abs(dense_layer.x)
            statistics["dense_edges"].set_mean_and_max_abs(dense_layer.edge_attr)
            statistics["target_normalized_new_displacement"].set_mean_and_max_abs(
                target_layer.normalized_new_displacement
            )

        for key in statistics.keys():
            statistics[key].finalaze_mean()

        for graph_data in cmh.get_tqdm(
            dataloader, config=self.config, desc="Calculating dataset variance"
        ):
            assert len(graph_data) == 1
            (dense_layer, sparse_layer), target_layer = graph_data[0]
            statistics["sparse_nodes"].set_variance(sparse_layer.x)
            statistics["sparse_edges"].set_variance(sparse_layer.edge_attr)
            statistics["multilayer_edges"].set_variance(sparse_layer.edge_attr_to_down)
            statistics["dense_nodes"].set_variance(dense_layer.x)
            statistics["dense_edges"].set_variance(dense_layer.edge_attr)
            statistics["target_normalized_new_displacement"].set_variance(
                target_layer.normalized_new_displacement
            )

        for key in statistics.keys():
            statistics[key].finalize_variance()

        self.device_count = saved_device_count

        return statistics

    def get_statistics_pandas(self):
        saved_device_count = self.device_count
        self.device_count = 1

        dataloader = get_dataloader(
            dataset=self,
            rank=0,
            world_size=1,
            batch_size=32,
            num_workers=0,
            shuffle=True,
            load_data=True,
        )

        sparse_nodes_data = None
        sparse_edges_data = None
        multilayer_edges_data = None
        dense_nodes_data = None
        dense_edges_data = None
        target_data = None

        for graph_data in cmh.get_tqdm(
            dataloader, config=self.config, desc="Calculating dataset statistics"
        ):
            assert len(graph_data) == 1
            example = graph_data[0]
            dense_layer = example[0][0]
            sparse_layer = example[0][1]
            target_layer = example[1]

            def cat(data, new_data):
                # new_data = torch.mean(new_data, axis=0, keepdim=True)
                if data is None:
                    return new_data
                return torch.cat((data, new_data))

            sparse_nodes_data = cat(sparse_nodes_data, sparse_layer.x)
            sparse_edges_data = cat(sparse_edges_data, sparse_layer.edge_attr)
            multilayer_edges_data = cat(multilayer_edges_data, sparse_layer.edge_attr_to_down)

            dense_nodes_data = cat(dense_nodes_data, dense_layer.x)
            dense_edges_data = cat(dense_edges_data, dense_layer.edge_attr)

            target_data = cat(target_data, target_layer.normalized_new_displacement)
            # count -= 1
            # if count == 0:
            #     break

        self.device_count = saved_device_count
        statistics = {
            "sparse_nodes": FeaturesStatisticsPandas(
                label="sparse_nodes",
                data=sparse_nodes_data,
                # columns=SceneInput.get_nodes_data_description_sparse(self.dimension)
            ),
            "sparse_edges": FeaturesStatisticsPandas(label="sparse_edges", data=sparse_edges_data),
            "multilayer_edges": FeaturesStatisticsPandas(
                label="multilayer_edges", data=multilayer_edges_data
            ),
            "dense_nodes": FeaturesStatisticsPandas(label="dense_nodes", data=dense_nodes_data),
            "dense_edges": FeaturesStatisticsPandas(label="dense_edges", data=dense_edges_data),
            "target_normalized_new_displacement": FeaturesStatisticsPandas(
                label="target_normalized_new_displacement", data=target_data
            ),
        }
        return statistics
        # edges_statistics = FeaturesStatistics(
        #     edges_data, #SceneInput.get_edges_data_description(self.dimension)
        # )
        # target_statistics = FeaturesStatistics(
        #     target_data, [f"exact_acceleration_{i}" for i in range(self.dimension)]
        # )

    def get_worker_data_range(self):
        worker_info = get_worker_info()
        total_id = self.rank * worker_info.num_workers + worker_info.id
        all_ids = self.world_size * worker_info.num_workers
        return range(total_id, self.data_count, all_ids)

    def load_data(self):
        if self.loaded_data is not None or not self.load_data_to_ram:
            return
        worker_data_range = self.get_worker_data_range()

        self.loaded_data = []
        worker_info = get_worker_info()
        data_tqdm = cmh.get_tqdm(
            iterable=worker_data_range,
            config=self.config,
            desc=f"Loading data to RAM (node {self.rank}, worker {worker_info.id + 1})",
            position=self.rank * worker_info.num_workers + worker_info.id,
        )
        with pkh.open_file_read(self.features_data_path) as file:
            for index in data_tqdm:
                example = pkh.load_byte_index(self.features_indices[index], file)
                self.loaded_data.append(example)

    def get_features_and_targets_data(self, index: int) -> GraphData:
        if self.loaded_data is not None:
            shifted_index = index % len(self.loaded_data)
            return self.loaded_data[shifted_index]
        with pkh.open_file_read(self.features_data_path) as file:
            return pkh.load_byte_index(byte_index=self.features_indices[index], data_file=file)

    def check_and_print(
        self, all_data_count, current_index, scene, step_tqdm, tqdm_description, current_time
    ):
        images_count = self.config.dataset_images_count
        if images_count is None:
            return
        plot_index_skip = 1 if all_data_count < images_count else int(all_data_count / images_count)
        relative_index = 1 if plot_index_skip == 0 else current_index % plot_index_skip
        if relative_index == 0:
            step_tqdm.set_description(f"{tqdm_description} - plotting index {current_index}")
            self.plot_data_scene(scene, current_index, self.images_directory, current_time)
        if relative_index == 1:
            step_tqdm.set_description(tqdm_description)

    def plot_data_scene(self, scene, filename, catalog, current_time):
        cmh.create_folders(catalog)
        extension = "png"  # pdf
        path = f"{catalog}/{filename}.{extension}"
        plot_setting(
            current_time=current_time,
            scene=scene,
            path=path,
            draw_detailed=True,
            extension=extension,
        )

    def generate_data_process(self, num_workers: int = 1, process_id: int = 0):
        pass

    def solve_and_prepare_scene(self, scene, forces, energy_functions, reduced_energy_functions):
        scene.prepare(forces)

        # scene.reduced.exact_acceleration, _ = self.solve_function(
        #     scene=scene.reduced,
        #     initial_a=scene.reduced.exact_acceleration,
        #     energy_functions=reduced_energy_functions,
        # )
        # scene.reduced.lifted_acceleration = scene.reduced.exact_acceleration

        # scene.exact_acceleration = scene.lower_acceleration_from_position(
        #     scene.reduced.exact_acceleration
        # )

        scene.exact_acceleration, _ = self.solve_function(
            scene=scene, initial_a=scene.exact_acceleration, energy_functions=energy_functions
        )
        scene.lifted_acceleration = scene.exact_acceleration

        scene.reduced.lifted_acceleration = scene.lift_acceleration_from_position(
            scene.exact_acceleration
        )
        scene.reduced.exact_acceleration = scene.reduced.lifted_acceleration

        return scene, scene.exact_acceleration

    def safe_save_scene(self, scene, data_path: str):
        scene_copy = copy.copy(scene)  ###
        scene_copy.prepare_to_save()
        pkh.append_data(
            data=scene_copy,
            data_path=data_path,
            lock=self.files_lock,
        )

    def _getitem_jax(self, index: int):
        def get_list(index):
            graph_data = self.get_features_and_targets_data(index)
            return [graph_data.layer_list, graph_data.target_data]

        if self.item_fn:
            raise ArgumentError
        return [get_list(index + i * len(self)) for i in range(self.device_count)]

    @property
    def _len_jax(self):
        # if not 'validation' in self.main_directory:
        #     return 50
        return self.data_count // self.device_count

    def _getitem_torch(self, index: int):
        # self.load_data()
        graph_data = self.get_features_and_targets_data(index)
        if self.item_fn:
            return self.item_fn([graph_data.layer_list, graph_data.target_data])
        return graph_data.layer_list, graph_data.target_data  # , graph_data.scene
        # return [*graph_data.layer_list, graph_data.target_data]

    @property
    def _len_torch(self):
        return self.data_count

    def __getitem__(self, index: int):
        return self._getitem_jax(index) if self.use_jax else self._getitem_torch(index)

    def __len__(self):
        return self._len_jax if self.use_jax else self._len_torch
