import argparse
import os

if __name__ == "__main__":
    jax_64 = False  # True

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "-1"
    # os.environ["JAX_PLATFORM_NAME"] = "cpu"
    # os.environ["JAX_DISABLE_JIT"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if jax_64:
        os.environ["JAX_ENABLE_X64"] = "1"
        print("JAX 64 BIT MODE")
    else:
        print("JAX 32 BIT MODE")


# import lovely_jax as lj
# import lovely_tensors as lt

# lt.monkey_patch()
# lj.monkey_patch()

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
from deep_conmech.graph.model_jax import GraphModelDynamicJax
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.graph.net_jax import CustomGraphNetJax
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
    all_validation_datasets = get_all_val_datasets(config=config, rank=0, world_size=1)
    for datasets in all_validation_datasets:
        datasets.initialize_data()

    train_dataset = get_train_dataset(config.td.dataset, config=config, rank=0, world_size=1)
    train_dataset.initialize_data()

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

    all_validation_datasets = get_all_val_datasets(config=config, rank=rank, world_size=world_size)
    for d in all_validation_datasets:
        d.initialize_data()
    all_print_datasets = scenarios.all_print(config.td)

    if config.use_jax:
        model = GraphModelDynamicJax(
            train_dataset=train_dataset,
            all_validation_datasets=all_validation_datasets,
            print_scenarios=all_print_datasets,
            config=config,
            rank=rank,
            world_size=world_size,
        )
    else:
        net = CustomGraphNet(statistics=statistics, td=config.td).to(rank)
        if config.load_newest_train:
            checkpoint_path = get_newest_checkpoint_path(config)
            net = GraphModelDynamic.load_checkpointed_net(net=net, rank=rank, path=checkpoint_path)

        model = GraphModelDynamic(
            train_dataset=train_dataset,
            all_validation_datasets=all_validation_datasets,
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
    all_print_scenaros = scenarios.all_print(config.td)

    if config.use_jax:
        checkpoint_path = get_newest_checkpoint_path(config)
        state = GraphModelDynamicJax.load_checkpointed_net(net=0, rank=0, path=checkpoint_path)
        GraphModelDynamicJax.plot_all_scenarios(state, all_print_scenaros, config)
    else:
        net = CustomGraphNet(statistics=statistics, td=config.td).to(0)
        checkpoint_path = get_newest_checkpoint_path(config)
        net = GraphModelDynamic.load_checkpointed_net(net=net, rank=0, path=checkpoint_path)
        GraphModelDynamic.plot_all_scenarios(net, all_print_scenaros, config)


def get_train_dataset(
    dataset_type,
    config: TrainingConfig,
    rank: int,
    world_size: int,
    item_fn=None,
):
    if dataset_type == "synthetic":
        train_dataset = SyntheticDataset(
            description="train",
            use_jax=config.use_jax,
            load_data_to_ram=config.load_training_data_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize=True,
            config=config,
            rank=rank,
            world_size=world_size,
            item_fn=item_fn,
        )
    elif dataset_type == "calculator":
        train_dataset = CalculatorDataset(
            description="train",
            use_jax=config.use_jax,
            all_scenarios=scenarios.all_train(config.td),
            load_data_to_ram=config.load_training_data_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize=True,
            config=config,
            rank=rank,
            world_size=world_size,
            item_fn=item_fn,
        )
    else:
        raise ValueError("Bad dataset type")
    return train_dataset


def get_all_val_datasets(config: TrainingConfig, rank: int, world_size: int):
    all_val_datasets = []
    for all_scenarios in scenarios.all_validation(config.td):
        description = "validation_" + str.join("/", [scenario.name for scenario in all_scenarios])
        all_val_datasets.append(
            CalculatorDataset(
                description=description,
                use_jax=config.use_jax,
                all_scenarios=all_scenarios,
                load_data_to_ram=config.load_validation_data_to_ram,
                with_scenes_file=False,
                randomize=False,
                config=config,
                rank=rank,
                world_size=world_size,
            )
        )
    return all_val_datasets


def get_newest_checkpoint_path(config: TrainingConfig):
    def get_index(path):
        return int(path.split("/")[-1].split(" ")[0])

    if config.use_jax:
        saved_model_paths = cmh.find_files_by_name(config.output_catalog, "checkpoint_0")
        return saved_model_paths[-1]

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
    print(f"Use JAX: {config.use_jax}")
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
