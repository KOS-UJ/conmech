import deep_conmech.scenarios as scenarios
from deep_conmech.graph.data.data_scenario import *
from deep_conmech.graph.data.data_synthetic import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.model import GraphModelDynamic
from deep_conmech.graph.net import CustomGraphNet
from conmech.helpers import helpers


def main():
    helpers.set_memory_limit()
    # torch.multiprocessing.set_start_method('spawn')
    # path = "output/10-22.57.40/16445595359197 - MODEL.pt"
    path = None

    nodes_statistics, edges_statistics = None, None  # train_dataset.get_statistics()
    net = CustomGraphNet(2, nodes_statistics, edges_statistics).to(thh.device)

    # train_dataset = TrainingSyntheticDatasetDynamic(dim=2)
    # train_dataset = TrainingScenariosDatasetDynamic(scenarios.all_train, Calculator.solve_all)
    train_dataset = TrainingScenariosDatasetDynamic(scenarios.all_train, net.solve_all)
    all_val_datasets = [
        ValidationScenarioDatasetDynamic(scenario)
        for scenario in scenarios.all_validation
    ]

    model = GraphModelDynamic(train_dataset, all_val_datasets, scenarios.all_print, net)
    if path is not None:
        model.load(path)
        model.print_raport()

    model.train()


if __name__ == "__main__":
    main()


# change names: base settings to body and setting obstacle to scene

# we use Lagrangian description (https://en.wikipedia.org/wiki/Continuum_mechanics#Lagrangian_description)


#####################################

# run_conmech_static()
# run_graph_static()

# train_forces_functions = [
#    examples.f_rotate,
#    examples.f_push,
#    examples.reverse(examples.f_rotate),
#    examples.reverse(examples.f_push)
# ]


# TODO:

# give network get_edges_features_list

# podawac v wymnozone przez ts

# wyswietlac dane < cutoff w getitem z nowymi randomizacjami


# print data from folder

# print based on validation dataset

# równo rozdzielić generowanie danych na workery (zamiast tego że każdy sprawdza)

# check if results are different if we set v_old, u_old instead of v_new, u_new in integral

# check conmech dirty - if it works with obstacle (add noise-correction to penetration?)


# in batch getittem randomly choose from original folder / folder  with generated settings


# set training set as validation - check if it will learn well
# decide between float and double
# add a_normalized_mean - make it work and check if it helps
# decide on imputs - function

# replace synthetic data with data produced by model


# add boundary conditions (as another graph or inside L2)
# add temperature

# add adaptivitiy prediction

# Add noise to points, U and V and set target to denoised setting - check if it makes sence with A.2.2
# Add Batchnorm at start
# remove self edgestime
# more message steps


##################


# do not randomize when printing and evaluating

# change loss function - sqrt of it (?)

# add virtual node
# https://sites.google.com/view/meshgraphnets
# ~2000 nodes
# 500 time steps

# normalize embeddings with L2 (check if LayerNorm does that )

# DRAWING:
# a * ts * ts
# v * ts
# u
# wszystko * scalar (10)


# przy liczniu u brać pod uwagę aktualne v * ts (podobnie z v (?))


# actual timestamp to file name

# multiple mesh sizes + pygmesh
# train on small domain, test on much bigger

# (Recurrent model - worse generalization)

# other body shapes

# spectral graphs

# this model gives us gradients - check

# adaptive mesh using another net

# dodać więcej parametrów do grafu - jak lambda, mu, typ cząstki itd.
