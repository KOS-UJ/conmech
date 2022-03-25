import deep_conmech.scenarios as scenarios
from deep_conmech.graph.data.data_scenario import *
from deep_conmech.graph.data.data_synthetic import *
from deep_conmech.graph.helpers import dch, thh
from deep_conmech.graph.model import GraphModelDynamic
from deep_conmech.graph.net import CustomGraphNet


def main():
    dch.set_memory_limit()
    # torch.multiprocessing.set_start_method('spawn')
    # path = "output/10-22.57.40/16445595359197 - MODEL.pt"
    path = None

    train_dataset = TrainingScenariosDatasetDynamic(
        scenarios.all_train, Calculator.solve_all
    )
    nodes_statistics, edges_statistics = train_dataset.get_statistics()
    #nodes_statistics, edges_statistics = None, None
    net = CustomGraphNet(2, nodes_statistics, edges_statistics).to(thh.device)

    # train_dataset = TrainingSyntheticDatasetDynamic(dimension=2)
    # train_dataset = TrainingScenariosDatasetDynamic(scenarios.all_train, net.solve_all, update_data=True)
    #all_val_datasets = [
    #    ValidationScenarioDatasetDynamic([scenario], scenario.id)
    #    for scenario in scenarios.all_validation
    #]
    #all_val_datasets.append(
    #    ValidationScenarioDatasetDynamic(scenarios.all_validation, "ALL")
    #)
    #val_stat = [dataset.get_statistics() for dataset in all_val_datasets]
    # nodes_statistics.describe()["forces_norm"]["mean"]
    # mean_val = np.mean(
    #    [
    #        val_stat[i][0].describe()["forces_norm"]["mean"]
    #        for i, _ in enumerate(scenarios.all_validation)
    #    ]
    # )
    model = GraphModelDynamic(
        train_dataset, all_val_datasets, scenarios.all_print(), net
    )
    if path is not None:
        model.load(path)
        model.print_raport()

    model.train()


if __name__ == "__main__":
    main()
