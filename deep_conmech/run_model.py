import argparse
from argparse import ArgumentParser, Namespace
from typing import Optional

from conmech.scenarios import scenarios
from deep_conmech.data.calculator_dataset import CalculatorDataset
from deep_conmech.data.dataset_statistics import DatasetStatistics
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.graph.model import GraphModelDynamic
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.helpers import dch, thh
from deep_conmech.training_config import TrainingConfig


def train(config: TrainingConfig):
    train_dataset = get_train_dataset(config.td.dataset, config=config)
    statistics = (
        train_dataset.get_statistics(layer_number=0) if config.td.use_dataset_statistics else None
    )
    net = get_net(statistics, config)

    all_val_datasets = get_all_val_datasets(config=config)
    all_print_datasets = scenarios.all_print(config.td)
    model = GraphModelDynamic(
        train_dataset=train_dataset,
        all_val_datasets=all_val_datasets,
        print_scenarios=all_print_datasets,
        net=net,
        config=config,
    )
    model.train()


def plot(config: TrainingConfig):
    if config.td.use_dataset_statistics:
        train_dataset = get_train_dataset(config.td.dataset, config=config)
        statistics = train_dataset.get_statistics(layer_number=0)
    else:
        statistics = None

    net = get_net(statistics, config)

    path = GraphModelDynamic.get_newest_saved_model_path(config)
    net.load(path)
    all_print_datasets = scenarios.all_print(config.td)
    GraphModelDynamic.plot_all_scenarios(net, all_print_datasets, config)


def get_train_dataset(dataset_type, config: TrainingConfig):
    if dataset_type == "synthetic":
        train_dataset = SyntheticDataset(
            description="train",
            layers_count=config.td.mesh_layers_count,
            load_features_to_ram=config.load_train_features_to_ram,
            load_targets_to_ram=config.load_train_targets_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize_at_load=True,
            config=config,
        )
    elif dataset_type == "calculator":
        train_dataset = CalculatorDataset(
            description="train",
            all_scenarios=scenarios.all_train(config.td),
            layers_count=config.td.mesh_layers_count,
            skip_index=1,
            load_features_to_ram=config.load_train_features_to_ram,
            load_targets_to_ram=config.load_train_targets_to_ram,
            randomize_at_load=True,
            config=config,
        )
    else:
        raise ValueError("Bad dataset type")
    return train_dataset


def get_all_val_datasets(config: TrainingConfig):
    all_val_datasets = []
    # if config.td.DATASET != "live":
    #    all_val_datasets.append(train_dataset)
    skip_index = 5
    # all_val_datasets.append(
    #     CalculatorDataset(
    #         description="val",
    #         all_scenarios=scenarios.all_validation(config.td),
    #         skip_index=skip_index,
    #         load_to_ram=False,
    #         config=config,
    #     )
    # )
    all_val_datasets.append(
        CalculatorDataset(
            description="all",
            all_scenarios=scenarios.all_train_and_validation(config.td),
            layers_count=config.td.mesh_layers_count,
            skip_index=skip_index,
            load_features_to_ram=False,
            load_targets_to_ram=False,
            randomize_at_load=False,
            config=config,
        )
    )
    # all_val_datasets.append(
    #    SyntheticDataset(description="train", dimension=2, load_to_ram=False, config=config)
    # )
    return all_val_datasets


def get_net(statistics: Optional[DatasetStatistics], config: TrainingConfig):
    net = CustomGraphNet(statistics=statistics, td=config.td)
    net.to(thh.device(config))
    return net


def main(args: Namespace):
    print(f"MODE: {args.mode}")
    device = thh.get_device_id()
    # dch.cuda_launch_blocking()
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)
    config = TrainingConfig(shell=args.shell, device=device)
    dch.set_memory_limit(config=config)
    print(f"Running using {config.device}")

    if "train" in args.mode:
        train(config)
    if args.mode == "plot":
        plot(config)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
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
