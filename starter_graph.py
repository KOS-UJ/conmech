import argparse
from argparse import ArgumentParser, Namespace

from deep_conmech.common.training_config import TrainingConfig
from deep_conmech.graph.data.data_scenario import *
from deep_conmech.graph.data.data_synthetic import *
from deep_conmech.graph.helpers import dch
from deep_conmech.graph.model import GraphModelDynamic
from deep_conmech.graph.net import CustomGraphNet


def get_train_dataset(dataset_type, config: TrainingConfig):
    if dataset_type == "synthetic":
        train_dataset = TrainingSyntheticDatasetDynamic(dimension=2, config=config)
    elif dataset_type == "scenarios":
        train_dataset = TrainingScenariosDatasetDynamic(
            scenarios.all_train(config.td), Solver.solve_all, config=config
        )
    else:
        raise ArgumentError("Bad dataset type")
    return train_dataset


def get_all_val_datasets(train_dataset, config: TrainingConfig):
    all_val_datasets = []
    all_val_datasets.append(train_dataset)
    all_val_datasets.append(
        ValidationScenarioDatasetDynamic(
            scenarios.all_validation(config.td), "ALL", config=config
        )
    )
    # all_val_datasets.extend(
    #    [
    #        ValidationScenarioDatasetDynamic([scenario], scenario.id)
    #        for scenario in scenarios.all_validation
    #    ]
    # )
    return all_val_datasets


def get_net_and_dataset(config: TrainingConfig):
    train_dataset = get_train_dataset(config.td.DATASET, config=config)
    statistics = train_dataset.get_statistics() if config.td.USE_DATASET_STATS else None
    net = CustomGraphNet(2, statistics=statistics, td=config.td)
    net.to(thh.device(config))
    return net, train_dataset


def train(config: TrainingConfig):
    net, train_dataset = get_net_and_dataset(config)
    all_val_datasets = get_all_val_datasets(train_dataset=train_dataset, config=config)
    model = GraphModelDynamic(train_dataset, all_val_datasets, net, config)
    model.train()


def plot(config: TrainingConfig):
    net, _ = get_net_and_dataset(config=config)
    path = GraphModelDynamic.get_newest_saved_model_path()
    net.load(path)
    all_print_datasets = scenarios.all_print(config.td)
    GraphModelDynamic.plot_all_scenarios(net, all_print_datasets, config)


def main(args: Namespace):
    print(f"MODE: {args.mode}")
    device = thh.get_device_id()
    config = TrainingConfig(SHELL=args.shell, DEVICE=device)
    dch.set_memory_limit(config=config)
    print(f"Running using {config.DEVICE}")

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

'''
import torch
x = torch.tensor([0, 1, 2, 3, 4])
a = torch.rand(2,3,4,5)
b = torch.zeros(37)
torch.save({"a": a, "b":b, "x", x}, 'tensors.pt')

#To append to a pickle file
import pickle

p={1:2}
q={3:4}
filename="picklefile"
with open(filename, 'a+') as fp:
    pickle.dump(p,fp)
    pickle.dump(q,fp)


#To load from pickle file
data = []
with open(filename, 'rb') as fr:
    try:
        while True:
            data.append(pickle.load(fr))
    except EOFError:
        pass
'''
