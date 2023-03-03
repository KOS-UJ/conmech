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
    input_batch_norm: bool = False

    @nn.compact
    def __call__(self, x, train):
        def kernel_init(key, shape, dtype=jnp.float_):
            return jax.random.uniform(key, shape, dtype, -1) / jnp.sqrt(x.shape[1])
            # in JAX code uniform is multiplied by 3, so using custom version

        def bias_init(key, shape, dtype=jnp.float_):
            return jax.random.uniform(key, shape, dtype, -1) / jnp.sqrt(x.shape[1])  # shape[0]
            # bias gets 64 features after linear layer (64) instead of input featureas (128)

        # _ = train
        if self.input_batch_norm:
            x = nn.BatchNorm(use_running_average=not train, momentum = 0.9, epsilon = 1e-8)(x)
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
    def propagate(
        self, node_latents_from, node_latents_to, edge_latents, edge_index, receivers_count
    ):
        senders, receivers = edge_index
        node_latents_senders = node_latents_from.at[senders].get()
        node_latents_receivers = node_latents_to.at[receivers].get()

        edge_inputs = self.get_edge_inputs(
            node_latents_senders, node_latents_receivers, edge_latents
        )

        new_edge_latents = self.message(edge_inputs=edge_inputs)

        aggregated_edge_latents = self.aggregate(
            new_edge_latents=new_edge_latents, receivers=receivers, receivers_count=receivers_count
        )

        new_node_latents = self.update(
            node_latents_to=node_latents_to,
            aggregated_edge_latents=aggregated_edge_latents,
        )
        return new_node_latents, edge_latents  # new_edge_latents

    def get_edge_inputs(self, node_latents_senders, node_latents_receivers, edge_latents):
        _ = node_latents_senders, node_latents_receivers
        return edge_latents

    def message(self, edge_inputs):
        return edge_inputs

    def aggregate(self, new_edge_latents, receivers, receivers_count):
        _ = receivers, receivers_count
        return new_edge_latents

    def update(self, node_latents_to, aggregated_edge_latents):
        _ = aggregated_edge_latents
        return node_latents_to


class ProcessorLayer(MessagePassingJax):
    latent_dimension: int
    internal_layer_count: int

    @nn.compact
    def __call__(self, node_latents, edge_latents, edge_index, receivers_count):
        new_node_latents, new_edge_latents = self.propagate(
            node_latents_from=node_latents,
            node_latents_to=node_latents,
            edge_latents=edge_latents,
            edge_index=edge_index,
            receivers_count=receivers_count,
        )
        return new_node_latents, new_edge_latents

    def get_edge_inputs(self, node_latents_senders, node_latents_receivers, edge_latents):
        edge_inputs = jnp.hstack((node_latents_senders, edge_latents, node_latents_receivers))
        return edge_inputs

    def message(self, edge_inputs):
        new_edge_latents = ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(edge_inputs, train=True)
        return new_edge_latents

    def aggregate(self, new_edge_latents, receivers, receivers_count):
        # alpha = self.attention(new_edge_latents, index)
        # TODO: check if sorted is needed, add degree normalizarion: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        aggregated_edge_latents = jax.ops.segment_sum(
            new_edge_latents, receivers, num_segments=receivers_count
        )
        return aggregated_edge_latents

    def update(self, node_latents_to, aggregated_edge_latents):
        node_inputs = jnp.hstack((node_latents_to, aggregated_edge_latents))
        new_node_latents = node_latents_to + ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(node_inputs, train=True)
        return new_node_latents


class LinkProcessorLayer(MessagePassingJax):
    latent_dimension: int
    internal_layer_count: int

    @nn.compact
    def __call__(
        self, node_latents_sparse, node_latents_dense, edge_latents_to_dense, edge_index_to_dense
    ):
        new_node_latents, _ = self.propagate(
            node_latents_from=node_latents_sparse,
            node_latents_to=node_latents_dense,
            edge_latents=edge_latents_to_dense,
            edge_index=edge_index_to_dense,
            receivers_count=None,
        )
        return new_node_latents

    def get_edge_inputs(self, node_latents_senders, node_latents_receivers, edge_latents):
        edge_inputs = jnp.hstack((node_latents_senders, edge_latents, node_latents_receivers))
        return edge_inputs

    def message(self, edge_inputs):
        new_edge_latents = ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(edge_inputs, train=True)
        return new_edge_latents

    def aggregate(self, new_edge_latents, receivers, receivers_count):
        _ = receivers, receivers_count
        latent_dim = new_edge_latents.shape[-1]
        # assuming special ordering (tested empirically)
        result = new_edge_latents.reshape(-1, CLOSEST_COUNT * latent_dim)
        return result

    def update(self, node_latents_to, aggregated_edge_latents):
        _ = node_latents_to
        linked_node_latents = ForwardNet(
            latent_dimension=self.latent_dimension, internal_layer_count=self.internal_layer_count
        )(aggregated_edge_latents, train=True)
        return linked_node_latents  # jnp.hstack((node_latents_to, linked_node_latents))


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
        latent_dimension = 64
        internal_layer_count = 1
        message_passes_sparse = 12  # 8
        message_passes_dense = 12 # 0  # 8
        dim = 3
        input_batch_norm = False  # True

        def propagate_messages(
            latent_dimension,
            node_latents,
            edge_latents,
            edge_index,
            receivers_count,
            message_passes,
        ):
            for _ in range(message_passes):
                node_latents, edge_latents = ProcessorLayer(
                    latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
                )(node_latents, edge_latents, edge_index, receivers_count)
            return node_latents

        def move_to_dense(
            latent_dimension,
            node_latents_sparse,
            node_latents_dense,
            edge_latents_to_dense,
            edge_index_to_dense,
        ):
            updated_node_latents_dense = LinkProcessorLayer(
                latent_dimension=latent_dimension, internal_layer_count=internal_layer_count
            )(
                node_latents_sparse=node_latents_sparse,
                node_latents_dense=node_latents_dense,
                edge_latents_to_dense=edge_latents_to_dense,
                edge_index_to_dense=edge_index_to_dense,
            )
            return updated_node_latents_dense

        node_latents_sparse = ForwardNet(
            latent_dimension=latent_dimension,
            internal_layer_count=internal_layer_count,
            input_batch_norm=input_batch_norm,
        )(args.sparse_x, train=train)

        edge_latents_sparse = ForwardNet(
            latent_dimension=latent_dimension,
            internal_layer_count=internal_layer_count,
            input_batch_norm=input_batch_norm,
        )(args.sparse_edge_attr, train=train)

        edge_latents_to_dense = ForwardNet(
            latent_dimension=latent_dimension,
            internal_layer_count=internal_layer_count,
            input_batch_norm=input_batch_norm,
        )(args.multilayer_edge_attr, train=train)

        node_latents_dense = ForwardNet(
            latent_dimension=latent_dimension,
            internal_layer_count=internal_layer_count,
            input_batch_norm=input_batch_norm,
        )(args.dense_x, train=train)

        edge_latents_dense = ForwardNet(
            latent_dimension=latent_dimension,
            internal_layer_count=internal_layer_count,
            input_batch_norm=input_batch_norm,
        )(args.dense_edge_attr, train=train)

        updated_node_latents_sparse = node_latents_sparse + propagate_messages(
            latent_dimension=latent_dimension,
            node_latents=node_latents_sparse,
            edge_latents=edge_latents_sparse,
            edge_index=args.sparse_edge_index,
            receivers_count=node_latents_sparse.shape[0],
            message_passes=message_passes_sparse,
        )

        # net_output_sparse = ForwardNet(
        #     latent_dimension=latent_dimension,
        #     internal_layer_count=internal_layer_count,
        #     output_linear_dim=dim,
        #     layer_norm=False,
        # )(updated_node_latents_sparse, train=True)

        # return net_output_sparse

        updated_node_latents_dense = move_to_dense(
            latent_dimension=latent_dimension,
            node_latents_sparse=updated_node_latents_sparse,
            node_latents_dense=node_latents_dense,
            edge_latents_to_dense=edge_latents_to_dense,
            edge_index_to_dense=args.multilayer_edge_index,
        )

        updated_node_latents_dense = updated_node_latents_dense + propagate_messages(
            latent_dimension=latent_dimension,
            node_latents=updated_node_latents_dense,
            edge_latents=edge_latents_dense,
            edge_index=args.dense_edge_index,
            receivers_count=updated_node_latents_dense.shape[0],
            message_passes=message_passes_dense,
        )

        net_output_dense = ForwardNet(
            latent_dimension=latent_dimension,
            internal_layer_count=internal_layer_count,
            output_linear_dim=dim,
            layer_norm=False,
        )(updated_node_latents_dense, train=True)

        batch_dummy = nn.BatchNorm(use_running_average=not train)(net_output_dense)
        return net_output_dense + 0 * batch_dummy

    def get_params(self, sample_args, init_rng):
        rngs_dict = {"params": init_rng}
        variables = self.init(rngs_dict, sample_args, train=False)
        return variables["params"], variables["batch_stats"]
