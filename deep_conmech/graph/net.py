import copy
from ctypes import ArgumentError
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.data.batch import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

from conmech.helpers import cmh, lnh, pkh
from conmech.helpers.cmh import DotDict
from conmech.mesh.mesh import Mesh
from conmech.solvers.calculator import Calculator
from deep_conmech.data.dataset_statistics import DatasetStatistics, FeaturesStatistics
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import CLOSEST_COUNT, TrainingData


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias, activation, dropout_rate):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_channels, out_channels, bias=bias))

        # if batch_norm:  # check also after ReLU
        #    layers.append(nn.BatchNorm1d(out_channels))

        if activation:
            layers.append(activation)

        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        output = self.blocks(x)
        return output


class ResidualBlock(nn.Module):
    class InternalResidualBlock(nn.Module):
        def __init__(self, channels, activation, dropout_rate):
            super().__init__()

            layers = []
            layers.append(nn.Linear(channels, channels))
            # if batch_norm:  # check also after ReLU
            #    layers.append(nn.BatchNorm1d(channels))

            layers.append(activation)

            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))

            self.blocks = nn.Sequential(*layers)

        def forward(self, x):
            output = self.blocks(x)
            return output

    def __init__(self, channels, activation, dropout_rate, skip):
        super().__init__()
        self.skip = skip

        self.blocks = nn.Sequential(
            self.InternalResidualBlock(
                channels=channels,
                # batch_norm=batch_norm,
                activation=activation,
                dropout_rate=dropout_rate,
            ),
            self.InternalResidualBlock(
                channels=channels,
                # batch_norm=batch_norm,
                activation=activation,
                dropout_rate=False,
            ),
        )

    def forward(self, x):
        output = self.blocks(x)
        if self.skip:
            output = x + output  # += not working on newer torch versions
        return output


class DataNorm(nn.Module):
    def __init__(self, in_channels, statistics: FeaturesStatistics):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("x_mean", statistics.data_mean)
        self.register_buffer("x_std", statistics.data_std)
        self.register_buffer("mask", statistics.data_std == 0)

    def forward(self, x):
        output = (x - self.x_mean) / self.x_std
        output = torch.nan_to_num(output)
        return output


class ForwardNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers_count: int,
        output_linear_dim: int,
        statistics: Optional[FeaturesStatistics],
        batch_norm: bool,
        layer_norm: bool,
        td: TrainingData,
    ):
        super().__init__()
        layers = []

        self.statistics = statistics
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
            if statistics is not None:
                raise ArgumentError()
        else:
            if statistics is not None:
                layers.append(DataNorm(in_channels=input_dim, statistics=statistics))

        layers.append(
            BasicBlock(
                in_channels=input_dim,
                out_channels=td.latent_dimension,
                bias=True,
                # batch_norm=config.BATCH_NORM,
                activation=td.activation,
                dropout_rate=False,
            )
        )

        for _ in range(layers_count):
            layers.append(
                ResidualBlock(
                    td.latent_dimension,
                    # batch_norm=config.BATCH_NORM,
                    activation=td.activation,
                    dropout_rate=td.dropout_rate,
                    skip=td.skip_connections,
                )
            )

        layers.append(
            BasicBlock(
                in_channels=td.latent_dimension,
                out_channels=output_linear_dim,
                bias=True,
                # batch_norm=False,
                activation=False,
                dropout_rate=False,
            )
        )

        if layer_norm:
            layers.append(nn.LayerNorm(output_linear_dim))

        self.net = thh.set_precision(nn.Sequential(*layers))

    def forward(self, x):
        result = self.net(x)
        return result


class Attention(nn.Module):
    def __init__(self, td: TrainingData):
        super().__init__()

        if td.attention_heads_count is None:
            self.blocks = None
            return

        attention_heads = BasicBlock(
            in_channels=td.latent_dimension,  # * 3,
            out_channels=td.attention_heads_count,
            bias=True,
            activation=td.activation,
            dropout_rate=False,
        )

        self.blocks = (
            attention_heads
            if td.attention_heads_count == 1
            else nn.Sequential(attention_heads, nn.Linear(td.attention_heads_count, 1, bias=False))
        )

    def forward(self, edge_inputs, index):
        if self.blocks is None:
            return 1.0

        alpha_score = self.blocks(edge_inputs)
        alpha = softmax(alpha_score, index)
        # torch.sum(alpha * (index == 5).reshape(-1,1)) == 1
        return alpha


# pylint: disable=W0223, W0221
class LinkProcessorLayerDown(MessagePassing):
    def __init__(self, td: TrainingData):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=td.latent_dimension * 2,
            layers_count=td.processor_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )
        self.attention = Attention(td=td)
        # self.epsilon = Parameter(torch.Tensor(1))

        self.decoder_inner = ForwardNet(
            input_dim=td.latent_dimension * CLOSEST_COUNT,
            layers_count=td.decoder_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=False, # TODO #65
            td=td,
        )

    def forward(self, edge_index, node_latents, edge_latents):
        processed_node_latents = self.propagate(
            edge_index=edge_index, node_latents=node_latents, edge_latents=edge_latents
        )
        return processed_node_latents

    def message(self, node_latents_i, node_latents_j, edge_latents):  # index
        edge_inputs = torch.hstack((node_latents_j, edge_latents))
        processed_edge_latents = self.edge_processor(edge_inputs)
        return processed_edge_latents

    def aggregate(self, new_edge_latents, index):  # weighted_edge_latents
        _ = index
        latent_dim = new_edge_latents.shape[-1]
        result = new_edge_latents.reshape(-1, CLOSEST_COUNT * latent_dim)
        return result

    def update(self, aggregated_edge_latents, node_latents):
        return self.decoder_inner(aggregated_edge_latents)

# pylint: disable=W0223, W0221
class LinkProcessorLayerUp(MessagePassing):
    def __init__(self, td: TrainingData):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=td.latent_dimension * 2,
            layers_count=td.processor_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )
        self.attention = Attention(td=td)
        # self.epsilon = Parameter(torch.Tensor(1))


    def forward(self, edge_index, node_latents, edge_latents):
        processed_node_latents = self.propagate(
            edge_index=edge_index, node_latents=node_latents, edge_latents=edge_latents
        )
        return processed_node_latents

    def message(self, node_latents_i, node_latents_j, edge_latents):  # index
        edge_inputs = torch.hstack((node_latents_j, edge_latents))
        processed_edge_latents = self.edge_processor(edge_inputs)
        return processed_edge_latents

    def aggregate(self, new_edge_latents, index):  # weight
        alpha = self.attention(new_edge_latents, index)
        aggregated_edge_latents = scatter_sum(alpha * new_edge_latents, index, dim=0)
        return aggregated_edge_latents

    def update(self, aggregated_edge_latents, node_latents):
        return torch.hstack((aggregated_edge_latents, node_latents[1]))
        to_node_latents = node_latents[-1]
        return torch.hstack((to_node_latents, aggregated_edge_latents))


# pylint: disable=W0223, W0221
class ProcessorLayer(MessagePassing):
    def __init__(self, td: TrainingData):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=td.latent_dimension * 2,
            layers_count=td.processor_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )
        self.node_processor = ForwardNet(
            input_dim=td.latent_dimension * 2,
            layers_count=td.processor_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )
        self.attention = Attention(td=td)

        # self.epsilon = Parameter(torch.Tensor(1))
        self.new_edge_latents = None

    def forward(self, edge_index, node_latents, edge_latents):
        self.new_edge_latents = None
        new_node_latents = self.propagate(
            edge_index=edge_index, node_latents=node_latents, edge_latents=edge_latents
        )
        return new_node_latents, edge_latents  # self.new_edge_latents

    def message(self, node_latents_i, node_latents_j, edge_latents):  # index
        edge_inputs = torch.hstack((node_latents_j, edge_latents))
        self.new_edge_latents = self.edge_processor(edge_inputs)
        return self.new_edge_latents

    def aggregate(self, new_edge_latents, index):  # weighted_edge_latents
        alpha = self.attention(new_edge_latents, index)
        aggregated_edge_latents = scatter_sum(alpha * new_edge_latents, index, dim=0)
        return aggregated_edge_latents

    def update(self, aggregated_edge_latents, node_latents):
        node_inputs = torch.hstack((node_latents, aggregated_edge_latents))
        new_node_latents = node_latents + self.node_processor(node_inputs)
        return new_node_latents


class CustomGraphNet(nn.Module):
    def __init__(
        self,
        statistics: Optional[DatasetStatistics],
        td: TrainingData,
    ):
        super().__init__()
        self.td = td

        self.node_encoder_sparse = ForwardNet(
            input_dim=SceneInput.get_nodes_data_up_dim(td.dimension),
            layers_count=td.encoder_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None if statistics is None else statistics.nodes_statistics,
            batch_norm=td.input_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )
        self.node_encoder_dense = ForwardNet(
            input_dim=SceneInput.get_nodes_data_down_dim(td.dimension),
            layers_count=td.encoder_layers_count,
            output_linear_dim=td.latent_dimension,  # 1 2
            statistics=None if statistics is None else statistics.nodes_statistics,
            batch_norm=td.input_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )

        self.edge_encoder = ForwardNet(
            input_dim=SceneInput.get_edges_data_dim(td.dimension),
            layers_count=td.encoder_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None if statistics is None else statistics.edges_statistics,
            batch_norm=td.input_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )

        self.multilayer_edge_encoder = ForwardNet(
            input_dim=SceneInput.get_multilayer_edges_data_dim(td.dimension),
            layers_count=td.encoder_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None if statistics is None else statistics.edges_statistics,
            batch_norm=td.input_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )

        #mesh_layers_count = 2
        self.processor_layers = nn.ModuleList(
            [ProcessorLayer(td=td) for _ in range(td.message_passes)] # td.message_passes * (mesh_layers_count * 2 - 1))]
        )
        self.downward_processor_layer = LinkProcessorLayerDown(td=td)
        self.upward_processor_layer = LinkProcessorLayerUp(td=td)

        self.decoder_dense = ForwardNet(
            input_dim=td.latent_dimension,
            layers_count=td.decoder_layers_count,
            output_linear_dim=td.dimension * 2,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=False,  # TODO #65
            td=td,
        )
        self.decoder_sparse = ForwardNet(
            input_dim=td.latent_dimension * 2,
            layers_count=td.decoder_layers_count,
            output_linear_dim=td.dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=False,  # TODO #65
            td=td,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def statistics(self):
        return self.statistics

    @property
    def node_statistics(self):
        return self.node_encoder.statistics

    @property
    def edge_statistics(self):
        return self.edge_encoder.statistics

    def move_to_dense(self, node_latents_sparse, node_latents_dense, edge_latents, edge_index):
        new_node_latents = self.downward_processor_layer(
            edge_index=edge_index,
            node_latents=(node_latents_sparse, node_latents_dense),
            edge_latents=edge_latents,
        )
        # residual connection (included in processor)
        # new_node_latents = node_latents + node_latents_from_up
        return new_node_latents
    
    def move_to_sparse(self, node_latents_sparse, node_latents_dense, edge_latents, edge_index):
        new_node_latents = self.upward_processor_layer(
            edge_index=edge_index,
            node_latents=(node_latents_dense, node_latents_sparse),
            edge_latents=edge_latents,
        )
        return new_node_latents

    def propagate_messages(
        self, layer: Data, node_latents: torch.Tensor, edge_latents: torch.Tensor
    ):
        for _ in range(self.td.message_passes):
            node_latents, edge_latents = self.processor_layers[self.processor_number](
                layer.edge_index, node_latents, edge_latents
            )
            self.processor_number += 1
        return node_latents

    def forward(self, layer_list: List[Data]):
        if isinstance(layer_list[0].x, Tuple):
            layer_list = [DotDict(l.x) for l in layer_list]
        self.processor_number = 0

        layer_dense = layer_list[0]
        node_latents_dense = self.node_encoder_dense(layer_dense["x"])
        edge_latents_dense = self.edge_encoder(layer_dense.edge_attr)

        layer_sparse = layer_list[1]
        node_latents_sparse = self.node_encoder_sparse(layer_sparse["x"])
        multilayer_edge_latents=self.multilayer_edge_encoder(layer_sparse.edge_attr_to_down)

        node_latents_common = self.move_to_dense(
            node_latents_sparse=node_latents_sparse,
            node_latents_dense=node_latents_dense,
            edge_latents=multilayer_edge_latents,
            edge_index=layer_sparse.edge_index_to_down,
        ) # torch.hstack((node_latents_dense, self.decoder_inner(node_latents_from_sparse)))
        node_latents_common = self.propagate_messages(
            layer=layer_dense, node_latents=node_latents_common, edge_latents=edge_latents_dense
        )

        edge_index_from_down = torch.flip(layer_sparse.edge_index_to_down, dims= (0,))
        node_latents_from_dense = self.move_to_sparse(
            node_latents_sparse=node_latents_sparse,
            node_latents_dense=node_latents_common,
            edge_latents=multilayer_edge_latents,
            edge_index=edge_index_from_down,
        )

        net_output_dense = self.decoder_dense(node_latents_common)
        net_output_sparse = self.decoder_sparse(node_latents_from_dense)

        return net_output_dense, net_output_sparse

    def solve(self, scene: SceneInput, initial_a):
        self.eval()
        # scene.linear_acceleration = Calculator.solve_acceleration_normalized_function(
        #     setting=scene, temperature=None, initial_a=initial_a
        # )

        scene.reduced.exact_acceleration = Calculator.solve(
            scene=scene.reduced, initial_a=scene.reduced.exact_acceleration
        )
        # return Calculator.solve(scene=scene, initial_a=initial_a)

        layers_list = [
            scene.get_features_data(layer_number=layer_number).to(self.device)
            for layer_number, _ in enumerate(scene.all_layers)
        ]

        net_result = self(layer_list=layers_list)
        net_result_dense, net_result_sparse = thh.to_np_double(net_result[0]), thh.to_np_double(net_result[1])  # + scene.linear_acceleration
        scaled_net_displacement = net_result_dense[:, : scene.dimension]
        net_dense_exact_acceleration = net_result_dense[:, scene.dimension :]
        net_sparse_exact_acceleration = net_result_sparse

        acceleration_position = scene.force_denormalize(net_dense_exact_acceleration)
        sparse_acceleration = scene.force_denormalize(net_sparse_exact_acceleration)
        #scene.reduced.exact_acceleration = sparse_acceleration

        # base = scene.moved_base
        # position = scene.position

        scene_copy = copy.deepcopy(scene.reduced)
        scene_copy.iterate_self(scene.reduced.exact_acceleration)
        # scene_copy = copy.deepcopy(scene)
        # scene_copy.iterate_self(acceleration_position)
        base = scene_copy.moved_base
        position = scene_copy.position

        net_displacement = scaled_net_displacement * scene.time_step
        new_displacement = scene.get_displacement(
            base=base, position=position, base_displacement=net_displacement
        )

        acceleration_displacement = scene.from_displacement(
            new_displacement
        )  # new_normalized_displacement

        return acceleration_displacement  # acceleration_displacement
