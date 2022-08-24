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
from tqdm import tqdm

import flax
from deep_conmech import run_model
from deep_conmech.data import base_dataset
from deep_conmech.training_config import TrainingConfig
from flax import linen as nn
from flax.training.train_state import TrainState




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
    layer_sparse = layer_list[1]
    layer_dense = layer_list[0]

    nodes_sparse = np.array(layer_sparse.x)
    nodes_dense = np.array(layer_dense.x)
    # edges_dense = np.array(layer_dense.edge_attr)
    # senders_dense = np.array(layer_dense.edge_index[0])
    # receivers_dense = np.array(layer_dense.edge_index[1])

    # graph_dense = jraph.GraphsTuple(
    #     nodes=nodes_dense,
    #     edges=edges_dense,
    #     n_node=len(nodes_dense),  # np.array(layer.num_nodes),
    #     n_edge=len(edges_dense),  # np.array(layer.num_edges),
    #     senders=senders_dense,
    #     receivers=receivers_dense,
    #     globals=None,
    # )

    nodes_multi = np.vstack((nodes_dense, nodes_sparse + len(nodes_dense)))
    edges_multi = np.array(layer_sparse.edge_attr_to_down)
    senders_multi = np.array(layer_sparse.edge_index_to_down[0] + len(nodes_dense))
    receivers_multi = np.array(layer_sparse.edge_index_to_down[1])

    graph_multi = jraph.GraphsTuple(
        nodes=nodes_multi,
        edges=edges_multi,
        n_node=len(nodes_multi),  # np.array(layer.num_nodes),
        n_edge=len(edges_multi),  # np.array(layer.num_edges),
        senders=senders_multi,
        receivers=receivers_multi,
        globals=None,
    )

    labels = np.array(target_data.exact_acceleration)
    # graphs = pad_graph_to_nearest_power_of_two(graphs)
    return graph_multi, labels



#######################

config = TrainingConfig(shell=False)
config.dataloader_workers = 0
train_dataset = run_model.get_train_dataset(config.td.dataset, config=config, rank=0, world_size=1)
train_dataset.initialize_data()
train_dataset.load_indices()
train_dataloader = base_dataset.get_train_dataloader(train_dataset, world_size=1, rank=0)

###############################


class EdgeEncoder(nn.Module):
    @nn.compact
    def __call__(self, x, train):
        x = nn.BatchNorm(use_running_average=not train)(x) #, name="bn_init"
        x = nn.Dense(features=12)(x)
        x = nn.Dropout(rate=0.1, deterministic=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(features=12)(x)
        return x

class GraphNet(nn.Module):
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, train: bool) -> jnp.DeviceArray:

        def update_edge_fn(edge_features, sender_node_features, receiver_node_features, globals_):
            x = EdgeEncoder()(edge_features, train =True)
            return x

        def aggregate_edges_for_nodes_fn(data, segment_ids, num_segments):
            return jraph.segment_sum(data, segment_ids, num_segments)

        def update_node_fn(
            node_features, aggregated_sender_edge_features, aggregated_receiver_edge_features, globals_
        ):
            return aggregated_receiver_edge_features

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
        graph = message_passing(graph)

        return graph.nodes

    def get_params(self, sample_graph, init_rng):
        params_init_rng, dropout_init_rng = jax.random.split(init_rng, 2)
        rngs_dict = {"params": params_init_rng, "dropout": dropout_init_rng}
        variables = self.init(rngs_dict, sample_graph, train=False)
        return variables["params"], variables["batch_stats"]


#########################


def forward(apply_fn, params, data, batch_stats, dropout_rng, train: bool):
    variables = {"params": params, "batch_stats": batch_stats}
    rngs = {"dropout": dropout_rng} if train else None
    result, non_trainable_params = apply_fn(
        variables, data, rngs=rngs, mutable=["batch_stats"], train=train
    )
    new_batch_stats = non_trainable_params["batch_stats"]
    return result, new_batch_stats


@jax.jit
def train_step(state, graphs, labels, dropout_rng):
    def loss_fn(params):
        result, new_batch_stats = forward(
            apply_fn=state.apply_fn,
            params=params,
            batch_stats=state.batch_stats,
            data=graphs,
            dropout_rng=dropout_rng,
            train=True,
        )
        loss = jnp.linalg.norm(result)
        return loss, new_batch_stats

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(
        grads=grads, batch_stats=new_batch_stats
    )
    return state, loss


class NetState(TrainState):
    batch_stats: float

def create_train_state(learning_rate, sample_graph, init_rng):
    params, batch_stats = GraphNet().get_params(sample_graph, init_rng)
    optimizer = optax.adam(learning_rate=learning_rate)
    ts = NetState.create(apply_fn=GraphNet().apply, params=params, tx=optimizer, batch_stats=batch_stats)
    return ts



def train_one_epoch(state, dataloader, dropout_rng, sample_graph):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    batch_tqdm = tqdm(train_dataloader)
    for batch in batch_tqdm:
        #graphs, labels = prepare_graph_tuples(batch)
        graphs, labels = sample_graph, None
        state, loss = train_step(state, graphs, labels, dropout_rng)
        batch_tqdm.set_description(f"Loss: {loss}")

    return state, new_dropout_rng


learning_rate = 0.01
num_epochs = 20


main_rng = jax.random.PRNGKey(0)
init_rng, dropout_rng = jax.random.split(main_rng, 2)
sample_graph = prepare_graph_tuples(next(iter(train_dataloader)))[0]

train_state = create_train_state(learning_rate, sample_graph, init_rng)

for epoch in range(1, num_epochs + 1):
    train_state, dropout_rng = train_one_epoch(
        train_state, train_dataloader, dropout_rng, sample_graph
    )

