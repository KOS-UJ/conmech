import argparse

import deep_conmech.scenarios as scenarios
from deep_conmech.graph import graph_scenarios
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
    # all_val_datasets.append(
    #    ValidationScenarioDatasetDynamic(scenarios.all_validation, "ALL")
    # )
    # all_val_datasets.extend(
    #    [
    #        ValidationScenarioDatasetDynamic([scenario], scenario.id)
    #        for scenario in scenarios.all_validation
    #    ]
    # )
    return all_val_datasets


def get_net():
    train_dataset = get_train_dataset(training_config.DATASET)
    # nodes_statistics, edges_statistics = train_dataset.get_statistics()
    nodes_statistics, edges_statistics = None, None
    net = CustomGraphNet(2, nodes_statistics, edges_statistics).to(dch.DEVICE)
    return net, train_dataset


def train():
    net, train_dataset = get_net()
    # else:
    #    net = CustomGraphNet(2, None, None).to(dch.DEVICE)
    #    train_dataset = TrainingScenariosDatasetDynamic(scenarios.all_train, net.solve_all, update_data=True)
    all_val_datasets = get_all_val_datasets(train_dataset=train_dataset)
    model = GraphModelDynamic(train_dataset, all_val_datasets, net)
    model.train()


def plot():
    net, _ = get_net()
    path = GraphModelDynamic.get_newest_saved_model_path()
    net.load(path)
    GraphModelDynamic.plot_all_scenarios(
        net, graph_scenarios.all_print()
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
