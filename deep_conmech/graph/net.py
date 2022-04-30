from ctypes import ArgumentError
from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

from conmech.scene.scene import Scene
from deep_conmech.data.dataset_statistics import DatasetStatistics, FeaturesStatistics
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingData


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
            in_channels=td.latent_dimension * 3,
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
class ProcessorLayer(MessagePassing):
    def __init__(self, attention: Attention, td: TrainingData):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=td.latent_dimension * 3,
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
        self.attention = attention

        self.epsilon = Parameter(torch.Tensor(1))
        self.new_edge_latents = None

        # change heads to a
        # self.bias = Parameter(torch.Tensor(out_channels))
        # self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

    def forward(self, batch, node_latents, edge_latents):
        self.new_edge_latents = None
        return self.propagate(
            edge_index=batch.edge_index,
            node_latents=node_latents,
            edge_latents=edge_latents,
        )

    def message(self, node_latents_i, node_latents_j, edge_latents, index):
        edge_inputs = torch.hstack((node_latents_i, node_latents_j, edge_latents))
        self.new_edge_latents = edge_latents + self.edge_processor(edge_inputs)

        alpha = self.attention(edge_latents, index)
        weighted_edge_latents = alpha * self.new_edge_latents
        return weighted_edge_latents

    def aggregate(self, weighted_edge_latents, index):
        aggregated_edge_latents = scatter_sum(weighted_edge_latents, index, dim=0)
        return aggregated_edge_latents

    def update(self, aggregated_edge_latents, node_latents):
        # node_inputs = aggregated_edge_latents
        # node_inputs = ((1 + self.epsilon) * node_latents) + aggregated_edge_latents))
        node_inputs = torch.hstack((node_latents, aggregated_edge_latents))
        new_node_latents = node_latents + self.node_processor(node_inputs)
        return new_node_latents, self.new_edge_latents


class CustomGraphNet(nn.Module):
    def __init__(
        self,
        statistics: Optional[DatasetStatistics],
        td: TrainingData,
    ):
        super().__init__()
        self.td = td

        self.node_encoder = ForwardNet(
            input_dim=SceneInput.nodes_data_dim(td.dimension),
            layers_count=td.encoder_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None if statistics is None else statistics.nodes_statistics,
            batch_norm=td.input_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )

        self.edge_encoder = ForwardNet(
            input_dim=SceneInput.edges_data_dim(td.dimension),
            layers_count=td.encoder_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None if statistics is None else statistics.edges_statistics,
            batch_norm=td.input_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )

        self.attention = Attention(td=td)

        self.processor_layers = nn.ModuleList(
            [ProcessorLayer(attention=self.attention, td=td) for _ in range(td.message_passes)]
        )

        self.decoder = ForwardNet(
            input_dim=td.latent_dimension,
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

    def forward(self, batch):
        node_input = batch.x  # position "pos" will not generalize
        edge_input = batch.edge_attr

        node_latents = self.node_encoder(node_input)
        edge_latents = self.edge_encoder(edge_input)

        for processor_layer in self.processor_layers:
            node_latents, edge_latents = processor_layer(batch, node_latents, edge_latents)

        net_output = self.decoder(node_latents)
        return net_output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def solve_all(self, scene: SceneInput):
        self.eval()

        batch = scene.get_data(config=None)[0].to(self.device)
        normalized_a_cuda = self(batch)

        normalized_a = thh.to_np_double(normalized_a_cuda)
        a = scene.denormalize_rotate(normalized_a)
        return a, normalized_a

    def solve(self, scene: SceneInput, initial_a):
        _ = initial_a
        a, _ = self.solve_all(scene)
        return a
