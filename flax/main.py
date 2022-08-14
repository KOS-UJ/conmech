from typing import Callable, List

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import torch
import torch_geometric.data
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import flax
from deep_conmech import run_model
from deep_conmech.data import base_dataset
from deep_conmech.data.calculator_dataset import CalculatorDataset
from deep_conmech.training_config import TrainingConfig
from flax import linen as nn
from flax.training import train_state

dataset = TUDataset(root="datasets/ENZYMES", name="ENZYMES")


class ToyDataset(Dataset):
    def __init__(self, dataset):
        """Initializes the data reader by loading in data."""
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @property
    def config(self):
        return self.dataset.config

    def __getitem__(self, idx):
        sample = self.dataset[idx][0][0]

        data = torch_geometric.data.Data(
            node_attr=sample.x,
            edge_attr=sample.edge_attr,
            num_nodes=sample.num_nodes,
            n_node=sample.num_nodes,
            n_edge=sample.num_edges,
            senders=sample.edge_index[0],
            receivers=sample.edge_index[1],
            y=sample.y,
            globals=torch.ones([sample.num_nodes, 1]),
        )

        # sample = self.dataset[idx]
        # data = torch_geometric.data.Data(
        #     node_attr=sample.x,
        #     edge_attr=torch.ones((sample.num_edges, 3)),
        #     num_nodes=sample.num_nodes,
        #     n_node=sample.num_nodes,
        #     n_edge=sample.num_edges,
        #     senders=sample.edge_index[0],
        #     receivers=sample.edge_index[1],
        #     y=sample.y,
        #     globals=torch.ones([sample.num_nodes, 1]),
        # )

        return data


train_dataset_ = dataset[:540]
train_dataset = ToyDataset(train_dataset_)

print(f"Number of training graphs: {len(train_dataset_)}")


# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#######################################

config = TrainingConfig(shell=False)
config.dataloader_workers = 0
train_dataset = run_model.get_train_dataset(config.td.dataset, config=config, rank=0, world_size=1)
train_dataset.initialize_data()
train_dataset.load_indices()
# train_dataset = ToyDataset(train_dataset)
train_dataloader = base_dataset.get_train_dataloader(train_dataset, world_size=1, rank=0)

#######################################


# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def _nearest_bigger_power_of_two(x: int) -> int:
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
    pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)


def prepare_graph_tuples(batch):
    layer = batch[0][0]
    target = batch[-1]

    graphs = jraph.GraphsTuple(
        nodes=np.array(layer.x),
        edges=np.array(layer.edge_attr),
        n_node=np.array(layer.num_nodes),
        n_edge=np.array(layer.num_edges),
        senders=np.array(layer.edge_index[0]),
        receivers=np.array(layer.edge_index[1]),
        globals=None,
    )

    labels = np.array(target.exact_acceleration)
    # graphs = pad_graph_to_nearest_power_of_two(graphs)
    return graphs, labels


###############################


def network_definition(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    def update_edge_fn(edge_features, sender_node_features, receiver_node_features, globals_):
        return edge_features

    def aggregate_edges_for_nodes_fn(data, segment_ids, num_segments):
        return jraph.segment_sum(data, segment_ids, num_segments)

    def update_node_fn(
        node_features, aggregated_sender_edge_features, aggregated_receiver_edge_features, globals_
    ):
        # del aggregated_sender_edge_features
        return node_features  # (node_features + aggregated_receiver_edge_features) / 2

    # def aggregate_nodes_for_globals_fn(data, segment_ids, num_segments):
    #     return jraph.segment_sum(data, segment_ids, num_segments)

    # def update_globals_fn(aggregated_node_features, aggregated_edge_features, globals_):
    #     return globals_

    # def aggregate_edges_for_globals_fn(data, segment_ids, num_segments):
    #     return jraph.segment_sum(data, segment_ids, num_segments)

    message_passing = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
        update_node_fn=update_node_fn,
        aggregate_nodes_for_globals_fn=None,
        update_global_fn=None,
        aggregate_edges_for_globals_fn=None,
        attention_logit_fn=None,
        attention_reduce_fn=None,
    )

    for _ in range(10):
        graph = message_passing(graph)

    graph = graph._replace(nodes=graph.nodes[..., [1]])
    return graph


jitted_network_definition = jax.jit(network_definition)

#########################


def calculate_loss(self, batch_data: List[Data]):
    dimension = self.config.td.dimension
    batch_layers = batch_data[0][: self.config.td.mesh_layers_count]
    layer_list = [layer.to(self.rank, non_blocking=True) for layer in batch_layers]
    target_data = batch_data[1].to(self.rank, non_blocking=True)
    batch_main_layer = layer_list[0]
    graph_sizes_base = get_graph_sizes(batch_main_layer)
    node_features = batch_main_layer.x  # .to("cpu")

    all_predicted_normalized_a = self.ddp_net(layer_list)  # .to("cpu")
    all_acceleration = clean_acceleration(
        cleaned_a=all_predicted_normalized_a, a_correction=target_data.a_correction
    )

    loss_tuple = self.calculate_loss_all(
        dimension=dimension,
        node_features=node_features,
        target_data=target_data,
        all_acceleration=all_acceleration,
        graph_sizes_base=graph_sizes_base,
        all_exact_acceleration=target_data.exact_acceleration,
        all_linear_acceleration=None,  # target_data.linear_acceleration,
    )
    # acceleration_list = [*all_acceleration.detach().split(graph_sizes_base)]

    return loss_tuple  # *, acceleration_list

for step, batch in enumerate(train_dataloader):
    calculate_loss(batch)
    print(f"Step {step + 1}:")
    graphs, labels = prepare_graph_tuples(batch)
    processed_graphs = jitted_network_definition(graphs)
    result = processed_graphs.nodes
    # print('graphs: ', graphs)
    print()
