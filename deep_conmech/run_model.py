import argparse
import os
from argparse import ArgumentParser, Namespace
from ctypes import ArgumentError
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing

from conmech.helpers import cmh
from conmech.scenarios import scenarios
from deep_conmech.data.calculator_dataset import CalculatorDataset
from deep_conmech.data.dataset_statistics import DatasetStatistics
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.graph.model import GraphModelDynamic
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.helpers import thh
from deep_conmech.training_config import TrainingConfig


def setup_distributed(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def train(config: TrainingConfig):
    # Prepare dataset
    get_train_dataset(config.td.dataset, config=config, rank=0, world_size=1)

    if not config.distributed_training:
        train_single(config)
    else:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            dist_run,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )


def dist_run(
    rank: int,
    world_size: int,
    config: TrainingConfig,
):
    setup_distributed(rank=rank, world_size=world_size)
    train_single(config, rank=rank, world_size=world_size)
    cleanup_distributed()


def train_single(config, rank=0, world_size=1):
    train_dataset = get_train_dataset(
        config.td.dataset, config=config, rank=rank, world_size=world_size
    )
    statistics = (
        train_dataset.get_statistics(layer_number=0) if config.td.use_dataset_statistics else None
    )

    # all_val_datasets = get_all_val_datasets(config=config, rank=rank, world_size=world_size)
    all_val_datasets = None
    all_print_datasets = scenarios.all_print(config.td)

    train_dataset.load_data()
    # for dataset in all_val_datasets:
    #    dataset.load_data()

    net = CustomGraphNet(statistics=statistics, td=config.td).to(rank)
    model = GraphModelDynamic(
        train_dataset=train_dataset,
        all_val_datasets=all_val_datasets,
        print_scenarios=all_print_datasets,
        net=net,
        config=config,
        rank=rank,
        world_size=world_size,
    )
    if config.load_newest_train:
        path = get_newest_checkpoint_path(config)
        model.load_checkpoint(path=path)
    model.train()


def plot(config: TrainingConfig):
    if config.td.use_dataset_statistics:
        train_dataset = get_train_dataset(config.td.dataset, config=config)
        statistics = train_dataset.get_statistics(layer_number=0)
    else:
        statistics = None

    net = CustomGraphNet(statistics=statistics, td=config.td).to(rank)
    path = get_newest_checkpoint_path(config)
    net = GraphModelDynamic.load_checkpointed_net(net=net, path=path)

    all_print_datasets = scenarios.all_print(config.td)
    GraphModelDynamic.plot_all_scenarios(net, all_print_datasets, config)


def get_train_dataset(dataset_type, config: TrainingConfig, rank: int, world_size: int):
    if dataset_type == "synthetic":
        train_dataset = SyntheticDataset(
            description="train",
            layers_count=config.td.mesh_layers_count,
            load_features_to_ram=config.load_train_features_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize_at_load=True,
            config=config,
            rank=rank,
            world_size=world_size,
        )
    elif dataset_type == "calculator":
        train_dataset = CalculatorDataset(
            description="train",
            all_scenarios=scenarios.all_train_2(config.td),
            layers_count=config.td.mesh_layers_count,
            load_features_to_ram=config.load_train_features_to_ram,
            randomize_at_load=True,
            config=config,
            rank=rank,
            world_size=world_size,
        )
    else:
        raise ValueError("Bad dataset type")
    return train_dataset


def get_all_val_datasets(config: TrainingConfig, rank: int, world_size: int):
    all_val_datasets = []
    # if config.td.DATASET != "live":
    #    all_val_datasets.append(train_dataset)
    # all_val_datasets.append(
    #     CalculatorDataset(
    #         description="val",
    #         all_scenarios=scenarios.all_validation(config.td),,
    #         load_to_ram=False,
    #         config=config,
    #     )
    # )
    all_val_datasets.append(
        CalculatorDataset(
            description="all",
            all_scenarios=scenarios.all_train_2(config.td),
            layers_count=config.td.mesh_layers_count,
            load_features_to_ram=False,
            randomize_at_load=False,
            config=config,
            rank=rank,
            world_size=world_size,
        )
    )
    # all_val_datasets.append(
    #    SyntheticDataset(description="train", dimension=2, load_to_ram=False, config=config)
    # )
    return all_val_datasets


def get_newest_checkpoint_path(config: TrainingConfig):
    def get_index(path):
        return int(path.split("/")[-1].split(" ")[0])

    saved_model_paths = cmh.find_files_by_extension(config.output_catalog, "pt")
    if not saved_model_paths:
        raise ArgumentError("No saved models")

    newest_index = np.argmax(np.array([get_index(path) for path in saved_model_paths]))
    path = saved_model_paths[newest_index]

    print(f"Taking saved model {path.split('/')[-1]}")
    return path


def main(args: Namespace):
    print(f"MODE: {args.mode}")
    # dch.cuda_launch_blocking()
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)
    config = TrainingConfig(shell=args.shell)
    # dch.set_torch_sharing_strategy()
    # dch.set_memory_limit(config=config)
    print(f"Running using {config.device}")

    if "train" in args.mode:
        train(config)
    if args.mode == "plot":
        plot(config)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "plot"],
        default="train",
        help="Running mode of aplication",
    )
    parser.add_argument(
        "--shell", action=argparse.BooleanOptionalAction, default=False
    )  # Python 3.9+
    args = parser.parse_args()
    main(args)
