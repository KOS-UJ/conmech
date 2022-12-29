import copy
import random
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
        # kernel_init = jax.nn.initializers.variance_scaling(
        #     1.0 / 3.0, "fan_in", "uniform", in_axis=-2, out_axis=-1, batch_axis=(), dtype=jnp.float_
        # )  # in JAX code uniform is multiplied by 3

        def kernel_init(key, shape, dtype=jnp.float_):
            return jax.random.uniform(key, shape, dtype, -1) / jnp.sqrt(x.shape[1])

        def bias_init(key, shape, dtype=jnp.float_):
            return jax.random.uniform(key, shape, dtype, -1) / jnp.sqrt(x.shape[1])  # shape[0]
            # bias gets 64 features after linear layer (64) instead of input featureas (128)

        _ = train
        # if self.batch_norm:
        #     x = nn.BatchNorm(use_running_average=not train)(x)
        for _ in range(self.internal_layer_count):
            x = nn.Dense(
                features=self.latent_dimension, kernel_init=kernel_init, bias_init=bias_init
            )(x)
            x = nn.relu(x)  # nn.gelu(x)
        output_linear_dim = (
            self.output_linear_dim if self.output_linear_dim else self.latent_dimension
        )
        x = nn.Dense(features=output_linear_dim, kernel_init=kernel_init, bias_init=bias_init)(x)
        # if self.layer_norm:
        # x = nn.LayerNorm()(x)
        return x


class MessagePassingJax(nn.Module):
    def propagate(self, node_latents, edge_latents, edge_index, receivers_count):
        # scale = 1
        # TODO: Do only once?
        # receivers_count = edge_index[1].max() + 1
        # aggregated_edge_latents = jnp.zeros((receivers_count, edge_latents.shape[1]))

        senders = edge_index[0]
        # torch.index_select
        node_latents_j = node_latents.at[senders].get()
        edge_inputs = jnp.hstack((node_latents_j, edge_latents))
        # edge_inputs = jnp.hstack((node_latents_j / scale, edge_latents))
        new_edge_latents = self.message(edge_inputs=edge_inputs)
        aggregated_edge_latents = self.aggregate(
            new_edge_latents, edge_index, receivers_count=receivers_count
        )
        new_node_latents = self.update(
            node_latents=node_latents,
            aggregated_edge_latents=aggregated_edge_latents,
        )
        return new_node_latents, edge_latents  # new_edge_latents

    def message(self, node_latents_j, edge_latents, edge_index, receivers_count):
        pass

    def aggregate(self, new_edge_latents, edge_index, receivers_count):
        # alpha = self.attention(new_edge_latents, index)
        receivers_index = edge_index[1]

        # Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # TODO: check if sorted
        aggregated_edge_latents = jax.ops.segment_sum(
            new_edge_latents, receivers_index, num_segments=receivers_count  # TODO: Check
        )

        # aggregated_edge_latents2 = thh.convert_cuda_tensor_to_jax(
        #     scatter_sum(
        #         thh.convert_jax_cuda_tensor(new_edge_latents),
        #         torch.tensor(edge_index[1], dtype=torch.int64).cuda(),
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
    def __call__(self, node_latents, edge_latents, edge_index, receivers_count):
        new_node_latents, new_edge_latents = self.propagate(
            node_latents, edge_latents, edge_index, receivers_count
        )
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
            node_latents_sparse, edge_latents_to_dense, edge_index_to_dense, receivers_count=None
        )
        return new_node_latents_jax

    def message(self, edge_inputs):
        new_edge_latents = ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(edge_inputs, train=True)
        return new_edge_latents

    def aggregate(self, new_edge_latents, edge_index, receivers_count):
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

    sparse_edge_index: np.ndarray
    dense_edge_index: np.ndarray
    multilayer_edge_index: np.ndarray


class CustomGraphNetJax(nn.Module):
    @nn.compact
    def __call__(self, args: GraphNetArguments, train: bool):
        _ = train
        latent_dimension = 64
        internal_layer_count = 1
        message_passes = 8
        dim = 3

        def propagate_messages(node_latents, edge_latents, edge_index, receivers_count):
            # proc = ProcessorLayer()
            for _ in range(message_passes):
                node_latents, edge_latents = ProcessorLayer(
                    latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
                )(node_latents, edge_latents, edge_index, receivers_count)
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
            args.sparse_edge_index,
            receivers_count=node_latents_sparse.shape[0],
        )

        updated_node_latents_dense = move_to_dense(
            node_latents_sparse=updated_node_latents_sparse,
            edge_latents_to_dense=edge_latents_to_dense,
            edge_index_to_dense=args.multilayer_edge_index,
        )

        updated_node_latents_dense = updated_node_latents_dense + propagate_messages(
            updated_node_latents_dense,
            edge_latents_dense,
            args.dense_edge_index,
            receivers_count=updated_node_latents_dense.shape[0],
        )

        net_output_dense = ForwardNet(
            latent_dimension=latent_dimension,
            internal_layer_count=internal_layer_count,
            output_linear_dim=dim,
            layer_norm=False,
        )(updated_node_latents_dense, train=True)

        return net_output_dense

    def get_params(self, sample_args, init_rng):
        rngs_dict = {"params": init_rng}
        variables = self.init(rngs_dict, sample_args, train=False)
        return variables["params"]  # , variables["batch_stats"]
