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




config = TrainingConfig(shell=False)
config.dataloader_workers = 0
train_dataset = run_model.get_train_dataset(config.td.dataset, config=config, rank=0, world_size=1)
train_dataset.initialize_data()
train_dataset.load_indices()
# train_dataset = ToyDataset(train_dataset)
train_dataloader = base_dataset.get_train_dataloader(train_dataset, world_size=1, rank=0)


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
    layer_list = batch[0]
    target_data = batch[-1]
    layer = layer_list[0] #batch_main_layer

    graphs = [jraph.GraphsTuple(
        nodes=np.array(layer.x),
        edges=np.array(layer.edge_attr),
        n_node=np.array(layer.num_nodes),
        n_edge=np.array(layer.num_edges),
        senders=np.array(layer.edge_index[0]),
        receivers=np.array(layer.edge_index[1]),
        globals=None,
    ) for layer in layer_list]

    labels = np.array(target_data.exact_acceleration)
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


for step, batch in enumerate(train_dataloader):
    print(f"Step {step + 1}:")
    graphs, labels = prepare_graph_tuples(batch)
    
    # all_predicted_normalized_a = self.ddp_net(layer_list)

    processed_graphs = jitted_network_definition(graphs)
    result = processed_graphs.nodes
    # print('graphs: ', graphs)
    print()
