# from conmech.helpers.config import SET_ENV
# if name == "__main__":
#     SET_ENV()

import argparse
import os
from argparse import ArgumentParser, Namespace
from ctypes import ArgumentError

import jax
import netron
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
from dotenv import load_dotenv

from conmech.helpers import cmh, pca
from conmech.helpers.config import Config, SimulationConfig
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import bunny_fall_3d
from conmech.simulations import simulation_runner
from deep_conmech.data import base_dataset
from deep_conmech.data.calculator_dataset import CalculatorDataset
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.graph.model_jax import GraphModelDynamicJax, save_tf_model
from deep_conmech.graph.model_torch import GraphModelDynamicTorch
from deep_conmech.graph.net_jax import CustomGraphNetJax
from deep_conmech.graph.net_torch import CustomGraphNet
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


def get_device_count(config):
    if not config.use_jax:
        return 1
    return len(jax.local_devices())


def initialize_data(config: TrainingConfig):
    device_count = get_device_count(config)
    all_validation_datasets = get_all_val_datasets(
        config=config, rank=0, world_size=1, device_count=device_count  # 1
    )
    for datasets in all_validation_datasets:
        datasets.initialize_data()

    train_dataset = get_train_dataset(config.td.dataset, config=config, device_count=device_count)
    train_dataset.initialize_data()

    return train_dataset, all_validation_datasets


def train(config: TrainingConfig):
    train_dataset, all_validation_datasets = initialize_data(config=config)

    if not config.torch_distributed_training:
        train_single(
            config, train_dataset=train_dataset, all_validation_datasets=all_validation_datasets
        )
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


def train_single(config, rank=0, world_size=1, train_dataset=None, all_validation_datasets=None):
    device_count = get_device_count(config)
    if train_dataset is None:
        train_dataset = get_train_dataset(
            config.td.dataset,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
        )
        train_dataset.load_indices()

    statistics = (
        train_dataset.get_statistics(layer_number=0) if config.td.use_dataset_statistics else None
    )

    if all_validation_datasets is None:
        all_validation_datasets = get_all_val_datasets(
            config=config, rank=rank, world_size=world_size, device_count=device_count
        )
        for d in all_validation_datasets:
            d.initialize_data()

    all_print_datasets = scenarios.all_print(config.td, config.sc)

    if config.use_jax:
        model = GraphModelDynamicJax(
            train_dataset=train_dataset,
            all_validation_datasets=all_validation_datasets,
            print_scenarios=all_print_datasets,
            config=config,
        )
    else:
        net = CustomGraphNet(statistics=statistics, td=config.td).to(rank)
        if config.load_newest_train:
            checkpoint_path = get_newest_checkpoint_path(config)
            net = GraphModelDynamicTorch.load_checkpointed_net(
                net=net, rank=rank, path=checkpoint_path
            )

        model = GraphModelDynamicTorch(
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


def visualize(config: TrainingConfig):
    checkpoint_path = get_newest_checkpoint_path(config)
    dataset = get_train_dataset(config.td.dataset, config=config)
    dataset.initialize_data()

    if config.use_jax:
        model_path = "log/jax_model.tflite"
        state = GraphModelDynamicJax.load_checkpointed_net(path=checkpoint_path)
        save_tf_model(model_path, state, dataset)
    else:
        # Only weights
        model_path = checkpoint_path

        model_path = "log/torch_model.onnx"
        net = CustomGraphNet(statistics=None, td=config.td)
        net = GraphModelDynamicTorch.load_checkpointed_net(net=net, rank=0, path=checkpoint_path)

        # hstack is currently not supported
        # https://pytorch.org/docs/stable/onnx_supported_aten_ops.html
        # GraphModelDynamicTorch.save_onnx_model(model_path, net, dataset)

    netron.start(model_path)


def plot(config: TrainingConfig):
    if config.td.use_dataset_statistics:
        train_dataset = get_train_dataset(config.td.dataset, config=config)
        statistics = train_dataset.get_statistics(layer_number=0)
    else:
        statistics = None
    all_print_scenaros = scenarios.all_print(config.td, config.sc)

    if config.use_jax:
        checkpoint_path = get_newest_checkpoint_path(config)
        state = GraphModelDynamicJax.load_checkpointed_net(path=checkpoint_path)
        GraphModelDynamicJax.plot_all_scenarios(state, all_print_scenaros, config)
    else:
        net = CustomGraphNet(statistics=statistics, td=config.td).to(0)
        checkpoint_path = get_newest_checkpoint_path(config)
        net = GraphModelDynamicTorch.load_checkpointed_net(net=net, rank=0, path=checkpoint_path)
        GraphModelDynamicTorch.plot_all_scenarios(net, all_print_scenaros, config)


def run_pca(config: TrainingConfig):
    dataset = get_train_dataset(config.td.dataset, config=config)
    dataset.initialize_data()
    dataloader = base_dataset.get_train_dataloader(dataset)

    simulation_config = SimulationConfig(
        use_normalization=False,
        use_linear_solver=False,
        use_green_strain=True,
        use_nonconvex_friction_law=False,
        use_constant_contact_integral=False,
        use_lhs_preconditioner=False,
        use_pca=False,
    )
    final_time = 0.5
    all_scenarios = [
        scenarios.bunny_fall_3d(
            mesh_density=32,
            scale=1,
            final_time=final_time,
            simulation_config=simulation_config,
        ),
        scenarios.bunny_rotate_3d(
            mesh_density=32,
            scale=1,
            final_time=final_time,
            simulation_config=simulation_config,
        ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=True,
        config=Config(shell=False),
    )

    pca.run(dataloader)


def get_train_dataset(
    dataset_type,
    config: TrainingConfig,
    rank: int = 0,
    world_size: int = 1,
    device_count: int = 1,
    item_fn=None,
):
    device_count = get_device_count(config)
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
            device_count=device_count,
            item_fn=item_fn,
        )
    elif dataset_type == "calculator":
        train_dataset = CalculatorDataset(
            description="train",
            use_jax=config.use_jax,
            all_scenarios=scenarios.all_train(config.td, config.sc),
            load_data_to_ram=config.load_training_data_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize=True,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
            item_fn=item_fn,
        )
    else:
        raise ValueError("Wrong dataset type")
    return train_dataset


def get_all_val_datasets(config: TrainingConfig, rank: int, world_size: int, device_count: int):
    all_val_datasets = []
    for all_scenarios in scenarios.all_validation(config.td, config.sc):
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
                device_count=device_count,
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
    cmh.print_jax_configuration()
    print(f"MODE: {args.mode}, PID: {os.getpid()}")
    # dch.cuda_launch_blocking()
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)
    config = TrainingConfig(shell=args.shell)
    config.sc = SimulationConfig(
        use_normalization=False,
        use_linear_solver=False,
        use_green_strain=True,
        use_nonconvex_friction_law=False,
        use_constant_contact_integral=False,
        use_lhs_preconditioner=False,
        use_pca=False,
    )

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
    if args.mode == "visualize":
        visualize(config)
    if args.mode == "pca":
        run_pca(config)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")  # forkserver")
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "plot", "profile", "visualize", "pca"],
        default="plot",
        help="Running mode of aplication",
    )
    parser.add_argument(
        "--shell", action=argparse.BooleanOptionalAction, default=False
    )  # Python 3.9+
    args = parser.parse_args()
    # with jax.disable_jit():
    load_dotenv()
    main(args)
