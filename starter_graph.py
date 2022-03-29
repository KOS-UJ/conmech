import argparse

import deep_conmech.scenarios as scenarios
from deep_conmech.graph.data.data_scenario import *
from deep_conmech.graph.data.data_synthetic import *
from deep_conmech.graph import graph_scenarios
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
    # train_dataset = get_train_dataset(config.DATASET)
    nodes_statistics, edges_statistics = None, None
    # nodes_statistics, edges_statistics = train_dataset.get_statistics()
    net = CustomGraphNet(2, nodes_statistics, edges_statistics).to(thh.device)
    return net  # , train_dataset


def train():
    train_dataset = get_train_dataset(config.DATASET)
    net = get_net()
    # else:
    #    net = CustomGraphNet(2, None, None).to(thh.device)
    #    train_dataset = TrainingScenariosDatasetDynamic(scenarios.all_train, net.solve_all, update_data=True)
    all_val_datasets = get_all_val_datasets(train_dataset=train_dataset)
    model = GraphModelDynamic(train_dataset, all_val_datasets, net)
    model.train()


def plot():
    def get_index(path):
        return int(path.split('/')[-1].split(' ')[0])

    saved_model_paths = cmh.find_files_by_extension("output", "pt")
    if not saved_model_paths:
        print("No saved models")
        return
    newest_index = np.argmax(np.array([get_index(path) for path in saved_model_paths]))
    path = saved_model_paths[newest_index]

    print(f"Taking saved model {path.split('/')[-1]}")
    net = get_net()
    net.load(path)
    GraphModelDynamic.plot_all_scenarios(net, graph_scenarios.all_print())  # scenarios.all_print())


def main(mode):
    if mode=='train':
        train()
    if mode=='plot':
        plot()
        

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train','plot'])
    args = parser.parse_args()
    main(mode=args.mode)


