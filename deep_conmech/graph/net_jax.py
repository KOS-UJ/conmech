import copy
from ctypes import ArgumentError
from functools import partial
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
from flax import linen as nn
from torch_scatter import scatter_sum

from conmech.helpers import cmh, lnh, pkh
from conmech.mesh.mesh import Mesh
from conmech.scene.energy_functions import EnergyFunctions
from conmech.solvers.calculator import Calculator
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import CLOSEST_COUNT, TrainingData


class ForwardNet(nn.Module):
    latent_dimension: int
    internal_layer_count: int
    output_linear_dim: Optional[int] = None
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x, train):
        # kernel_init = jax.nn.initializers.lecun_normal()
        # kernel_init = jax.nn.initializers.kaiming_uniform()
        kernel_init = jax.nn.initializers.variance_scaling(
            0.4, "fan_in", "uniform", in_axis=-2, out_axis=-1, batch_axis=(), dtype=jnp.float_
        )

        # bias_init = jax.nn.initializers.zeros
        # bias_init = jax.nn.initializers.lecun_uniform(in_axis=-1)
        bias_init = jax.nn.initializers.variance_scaling(
            0.4, "fan_in", "uniform", in_axis=-1, out_axis=-1, batch_axis=(), dtype=jnp.float_
        )

        _ = train
        # if self.batch_norm:
        #     x = nn.BatchNorm(use_running_average=not train)(x)
        for _ in range(self.internal_layer_count):
            x = nn.Dense(
                features=self.latent_dimension, kernel_init=kernel_init, bias_init=bias_init
            )(x)
            x = nn.relu(x) #nn.gelu(x)
        output_linear_dim = (
            self.output_linear_dim if self.output_linear_dim else self.latent_dimension
        )
        x = nn.Dense(features=output_linear_dim, kernel_init=kernel_init, bias_init=bias_init)(x)
        # if self.layer_norm:
            # x = nn.LayerNorm()(x)
        return x


class MessagePassingJax(nn.Module):
    def propagate(self, node_latents, edge_latents, edge_index):
        scale = 1
        # TODO: Do only once?
        # receivers_count = edge_index[1].max() + 1
        # aggregated_edge_latents = jnp.zeros((receivers_count, edge_latents.shape[1]))

        senders = edge_index[0]
        # torch.index_select
        node_latents_j = node_latents.at[senders].get()
        edge_inputs = jnp.hstack((node_latents_j / scale, edge_latents))
        new_edge_latents = self.message(edge_inputs=edge_inputs)
        aggregated_edge_latents = self.aggregate(new_edge_latents, edge_index)
        new_node_latents = self.update(
            node_latents=node_latents, aggregated_edge_latents=aggregated_edge_latents
        )
        return new_node_latents, edge_latents  # new_edge_latents

    def message(self, node_latents_j, edge_latents, edge_index):
        pass

    def aggregate(self, new_edge_latents, edge_index):
        # alpha = self.attention(new_edge_latents, index)
        receivers_index = edge_index[1]
        receivers_count = receivers_index.max() + 1

        # receivers_count = node_latents_i.shape[0]

        # Equivalent to jnp.sum(n_node), but jittable
        # sum_n_node = tree.tree_leaves(nodes)[0].shape[0]

        # sent_attributes = jax.tree_util.tree_map(
        #   lambda e: jax.ops.segment_sum(e, edge_index[0], receivers_count), new_edge_latents)

        # aggregated_edge_latents = aggregated_edge_latents.at[edge_index[1]].add(new_edge_latents)
        
        # Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        #TODO: check if sorted
        aggregated_edge_latents = jax.ops.segment_sum(
            new_edge_latents, receivers_index, num_segments=receivers_count  # TODO: Check
        )

        # aggregated_edge_latents2 = thh.convert_cuda_tensor_to_jax(
        #     scatter_sum(
        #         thh.convert_jax_cuda_tensor(new_edge_latents),
        #         torch.tensor(edge_index[1]).cuda(),
        #         dim=0,
        #     )
        # )
        return aggregated_edge_latents

    def update(self, node_latents, aggregated_edge_latents):
        pass


class ProcessorLayer(MessagePassingJax):
    latent_dimension: int
    internal_layer_count: int

    @nn.compact
    def __call__(self, node_latents, edge_latents, edge_index):
        new_node_latents, new_edge_latents = self.propagate(node_latents, edge_latents, edge_index)
        return new_node_latents, new_edge_latents

    def message(self, edge_inputs):
        new_edge_latents = ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(edge_inputs, train=True)
        return new_edge_latents

    def update(self, node_latents, aggregated_edge_latents):
        node_inputs = jnp.hstack((node_latents, aggregated_edge_latents))
        new_node_latents = node_latents + ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(node_inputs, train=True)
        return new_node_latents


class LinkProcessorLayer(MessagePassingJax):
    latent_dimension: int
    internal_layer_count: int

    @nn.compact
    def __call__(self, node_latents_sparse, edge_latents_to_dense, edge_index_to_dense):
        new_node_latents_jax, _ = self.propagate(
            node_latents_sparse, edge_latents_to_dense, edge_index_to_dense
        )
        return new_node_latents_jax

    def message(self, edge_inputs):
        new_edge_latents = ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(edge_inputs, train=True)
        return new_edge_latents

    def aggregate(self, new_edge_latents, edge_index):
        _ = edge_index
        latent_dim = new_edge_latents.shape[-1]
        result = new_edge_latents.reshape(-1, CLOSEST_COUNT * latent_dim)
        return result

    def update(self, node_latents, aggregated_edge_latents):
        new_node_latents = ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(aggregated_edge_latents, train=True)
        return new_node_latents


class GraphNetArguments(NamedTuple):
    sparse_x: jnp.ndarray
    sparse_edge_attr: jnp.ndarray
    dense_x: jnp.ndarray
    dense_edge_attr: jnp.ndarray
    multilayer_edge_attr: jnp.ndarray


class GraphNetStaticArguments(NamedTuple):
    sparse_edge_index: np.ndarray
    dense_edge_index: np.ndarray
    multilayer_edge_index: np.ndarray

    # sparse_edge_index_count: int
    # dense_edge_index_count: int
    # multilayer_edge_index_count: int

    def __hash__(self):
        return 0  # jnp.sum(self.val))

    def __eq__(self, other):
        return isinstance(other, GraphNetStaticArguments) and jnp.all(jnp.equal(self.sparse_edge_index, other.sparse_edge_index))
    # TODO

class HashableArrayWrapper:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return 0  # jnp.sum(self.val))

    def __eq__(self, other):
        return isinstance(other, HashableArrayWrapper) and jnp.all(jnp.equal(self.val, other.val))

    def __jax_array__(self):
        return jnp.array(self.val)

class CustomGraphNetJax(nn.Module):
    def prepare_input(self, layer_list):
        def unpack(layer):
            return layer["x"], layer.edge_attr, layer.edge_index

        layer_dense = layer_list[0]
        layer_sparse = layer_list[1]

        dense_x, dense_edge_attr, dense_edge_index = unpack(layer_dense)
        sparse_x, sparse_edge_attr, sparse_edge_index = unpack(layer_sparse)
        multilayer_edge_attr = layer_sparse.edge_attr_to_down
        multilayer_edge_index = layer_sparse.edge_index_to_down

        # sparse_edge_index_count = int(sparse_edge_index[1].max()) + 1
        # multilayer_edge_index_count = int(multilayer_edge_index[1].max()) + 1
        # dense_edge_index_count = int(dense_edge_index[1].max()) + 1
      
        sparse_edge_index = HashableArrayWrapper(
            np.array(sparse_edge_index)
        )  # HashableArrayWrapper
        dense_edge_index = HashableArrayWrapper(np.array(dense_edge_index))
        multilayer_edge_index = HashableArrayWrapper(np.array(multilayer_edge_index))

        args = GraphNetArguments(
            sparse_x=sparse_x,
            sparse_edge_attr=sparse_edge_attr,
            dense_x=dense_x,
            dense_edge_attr=dense_edge_attr,
            multilayer_edge_attr=multilayer_edge_attr,
        )
        static_args = GraphNetStaticArguments(
            sparse_edge_index=sparse_edge_index,
            dense_edge_index=dense_edge_index,
            multilayer_edge_index=multilayer_edge_index,
            # sparse_edge_index_count=sparse_edge_index_count,
            # dense_edge_index_count=dense_edge_index_count,
            # multilayer_edge_index_count=multilayer_edge_index_count,
        )
        return args, static_args

    @nn.compact
    def __call__(self, args, static_args, train: bool):
        _ = train
        latent_dimension = 64
        internal_layer_count = 1
        message_passes = 8
        dim = 3

        def propagate_messages(node_latents, edge_latents, edge_index):
            # proc = ProcessorLayer()
            for _ in range(message_passes):
                node_latents, edge_latents = ProcessorLayer(
                    latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
                )(node_latents, edge_latents, edge_index)
            return node_latents

        def move_to_dense(node_latents_sparse, edge_latents_to_dense, edge_index_to_dense):
            updated_node_latents_dense = LinkProcessorLayer(
                latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
            )(
                node_latents_sparse,
                edge_latents_to_dense,
                edge_index_to_dense,
            )
            return updated_node_latents_dense

        node_latents_sparse = ForwardNet(
            latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
        )(args.sparse_x, train=True)

        edge_latents_sparse = ForwardNet(
            latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
        )(args.sparse_edge_attr, train=True)

        edge_latents_to_dense = ForwardNet(
            latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
        )(args.multilayer_edge_attr, train=True)

        node_latents_dense = ForwardNet(
            latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
        )(args.dense_x, train=True)

        edge_latents_dense = ForwardNet(
            latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
        )(args.dense_edge_attr, train=True)

        updated_node_latents_sparse = node_latents_sparse + propagate_messages(
            node_latents_sparse,
            edge_latents_sparse,
            static_args.sparse_edge_index.val,
        )

        updated_node_latents_dense = move_to_dense(
            node_latents_sparse=updated_node_latents_sparse,
            edge_latents_to_dense=edge_latents_to_dense,
            edge_index_to_dense=static_args.multilayer_edge_index.val,
        )

        updated_node_latents_dense = updated_node_latents_dense + propagate_messages(
            updated_node_latents_dense,
            edge_latents_dense,
            static_args.dense_edge_index.val,
        )

        net_output_dense = ForwardNet(
            latent_dimension=latent_dimension,
            internal_layer_count=internal_layer_count,
            output_linear_dim=dim,
            layer_norm=False
        )(updated_node_latents_dense, train=True)

        return net_output_dense

    def get_params(self, sample_args, sample_static_args, init_rng):
        rngs_dict = {"params": init_rng}
        variables = self.init(rngs_dict, sample_args, sample_static_args, train=False)
        return variables["params"]  # , variables["batch_stats"]

    def solve(self, scene: SceneInput, energy_functions: EnergyFunctions, initial_a):
        return Calculator.solve(scene=scene, energy_functions=energy_functions, initial_a=initial_a)
