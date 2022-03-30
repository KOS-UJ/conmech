import argparse

import deep_conmech.scenarios as scenarios
from deep_conmech.graph.data.data_scenario import *
from deep_conmech.graph.data.data_synthetic import *
from deep_conmech.graph.helpers import dch, thh
from deep_conmech.graph.model import GraphModelDynamic
from deep_conmech.graph.net import CustomGraphNet


def get_train_dataset(dataset_type):
    if dataset_type == "synthetic":
        train_dataset = TrainingSyntheticDatasetDynamic(dimension=2)
    elif dataset_type == "scenarios":
        train_dataset = TrainingScenariosDatasetDynamic(
            scenarios.all_train, Solver.solve_all
        )
    else:
        raise ArgumentError()
    return train_dataset


def get_all_val_datasets(train_dataset):
    all_val_datasets = []
    all_val_datasets.append(train_dataset)
    all_val_datasets.append(
       ValidationScenarioDatasetDynamic(scenarios.all_validation, "ALL")
    )
    # all_val_datasets.extend(
    #    [
    #        ValidationScenarioDatasetDynamic([scenario], scenario.id)
    #        for scenario in scenarios.all_validation
    #    ]
    # )
    return all_val_datasets


def get_net():
    statistics = None
    net = CustomGraphNet(2, statistics=statistics).to(dch.DEVICE)
    return net


def train():
    train_dataset = get_train_dataset(training_config.DATASET)
    # train_dataset = TrainingScenariosDatasetDynamic(scenarios.all_train, net.solve_all, update_data=True)
    # statistics = train_dataset.get_statistics()
    net = get_net()
    all_val_datasets = get_all_val_datasets(train_dataset=train_dataset)
    model = GraphModelDynamic(train_dataset, all_val_datasets, net)
    model.train()


def plot():
    net = get_net()
    path = GraphModelDynamic.get_newest_saved_model_path()
    net.load(path)
    GraphModelDynamic.plot_all_scenarios(
        net, scenarios.all_print()
    )


def main(args):
    print(f"MODE: {args.mode}")
    if "train" in args.mode:
        train()
    if args.mode == "plot":
        plot()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "plot"],
        default="train",
        help="Running mode of aplication",
    )
    '''
    parser.add_argument(
        "--shell",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Running in shell",
    )
    '''
    args = parser.parse_args()
    main(args)
