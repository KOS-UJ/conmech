import argparse
import os

import jax

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "-1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import socketserver
from argparse import ArgumentParser, Namespace
from ctypes import ArgumentError

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing

from conmech.helpers import cmh
from conmech.scenarios import scenarios
from deep_conmech.data.calculator_dataset import CalculatorDataset
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.graph.model import GraphModelDynamic
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.helpers import dch
from deep_conmech.training_config import TrainingConfig


def setup_distributed(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    # with socketserver.TCPServer(("localhost", 0), None) as s:
    #     free_port = str(s.server_address[1])
    free_port = "12348"
    os.environ["MASTER_PORT"] = free_port
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def train(config: TrainingConfig):
    train_dataset = get_train_dataset(config.td.dataset, config=config, rank=0, world_size=1)
    train_dataset.initialize_data()

    validation_dataset = get_val_dataset(config=config, rank=0, world_size=1)
    validation_dataset.initialize_data()

    if not config.distributed_training:
        train_single(config, train_dataset=train_dataset)
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


def train_single(config, rank=0, world_size=1, train_dataset=None):
    if train_dataset is None:
        train_dataset = get_train_dataset(
            config.td.dataset, config=config, rank=rank, world_size=world_size
        )
        train_dataset.load_indices()
    statistics = (
        train_dataset.get_statistics(layer_number=0) if config.td.use_dataset_statistics else None
    )

    validation_dataset = get_val_dataset(config=config, rank=rank, world_size=world_size)
    validation_dataset.load_indices()
    all_print_datasets = scenarios.all_print(config.td)

    net = CustomGraphNet(statistics=statistics, td=config.td).to(rank)
    if config.load_newest_train:
        checkpoint_path = get_newest_checkpoint_path(config)
        net = GraphModelDynamic.load_checkpointed_net(net=net, rank=rank, path=checkpoint_path)
    model = GraphModelDynamic(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        print_scenarios=all_print_datasets,
        net=net,
        config=config,
        rank=rank,
        world_size=world_size,
    )
    if config.load_newest_train:
        model.load_checkpoint(path=checkpoint_path)
    model.train()


def plot(config: TrainingConfig):
    if config.td.use_dataset_statistics:
        train_dataset = get_train_dataset(config.td.dataset, config=config)
        statistics = train_dataset.get_statistics(layer_number=0)
    else:
        statistics = None

    net = CustomGraphNet(statistics=statistics, td=config.td).to(0)
    checkpoint_path = get_newest_checkpoint_path(config)
    net = GraphModelDynamic.load_checkpointed_net(net=net, rank=0, path=checkpoint_path)

    all_print_datasets = scenarios.all_print(config.td)
    GraphModelDynamic.plot_all_scenarios(net, all_print_datasets, config)


def get_train_dataset(dataset_type, config: TrainingConfig, rank: int, world_size: int, item_fn=None):
    if dataset_type == "synthetic":
        train_dataset = SyntheticDataset(
            description="train",
            layers_count=config.td.mesh_layers_count,
            load_data_to_ram=config.load_training_data_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize_at_load=True,
            config=config,
            rank=rank,
            world_size=world_size,
            item_fn=item_fn
        )
    elif dataset_type == "calculator":
        train_dataset = CalculatorDataset(
            description="train",
            all_scenarios=scenarios.all_train(config.td),
            layers_count=config.td.mesh_layers_count,
            load_data_to_ram=config.load_training_data_to_ram,
            randomize_at_load=True,
            config=config,
            rank=rank,
            world_size=world_size,
            item_fn=item_fn
        )
    else:
        raise ValueError("Bad dataset type")
    return train_dataset


def get_val_dataset(config: TrainingConfig, rank: int, world_size: int):
    # all_val_datasets = []
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
    return CalculatorDataset(
        description="validation",
        all_scenarios=scenarios.all_validation(config.td),
        layers_count=config.td.mesh_layers_count,
        load_data_to_ram=config.load_validation_data_to_ram,
        randomize_at_load=False,
        config=config,
        rank=rank,
        world_size=world_size,
    )
    # )
    # all_val_datasets.append(
    #    SyntheticDataset(description="train", dimension=2, load_to_ram=False, config=config)
    # )
    # return all_val_datasets


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
    print(f"MODE: {args.mode}, PID: {os.getpid()}")
    # dch.cuda_launch_blocking()
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)
    config = TrainingConfig(shell=args.shell)
    # dch.set_torch_sharing_strategy()
    dch.set_memory_limit(config=config)
    print(f"Running using {config.device}")

    if args.mode == "train":
        train(config)
    if args.mode == "profile":
        config.max_epoch_number = 2
        train(config)
    if args.mode == "plot":
        plot(config)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")  # forkserver")
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "plot", "profile"],
        default="train",
        help="Running mode of aplication",
    )
    parser.add_argument(
        "--shell", action=argparse.BooleanOptionalAction, default=False
    )  # Python 3.9+
    args = parser.parse_args()
    # with jax.disable_jit():
    main(args)
