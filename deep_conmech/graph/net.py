from argparse import ArgumentError
from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

from deep_conmech.common.training_config import TrainingData
from deep_conmech.graph.data.dataset_statistics import (
    DatasetStatistics,
    FeaturesStatistics,
)
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting.setting_input import SettingInput

# TODO: move
ACTIVATION = nn.ReLU()  # nn.PReLU()  # ReLU


# | ac {.ACTIVATION._get_name()} \


def device(module: nn.Module):
    return next(module.parameters()).device


# next(net.edge_encoder.children())[0]


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.dropout_rate = dropout_rate


class BasicBlock(Block):
    def __init__(self, in_channels, out_channels, bias, activation, dropout_rate):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
        )
        self.activation = activation

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


class ResidualBlock(Block):
    class InternalResidualBlock(Block):
        def __init__(self, channels, dropout_rate):
            super().__init__(
                in_channels=channels, out_channels=channels, dropout_rate=dropout_rate,
            )

            layers = []
            layers.append(nn.Linear(channels, channels))
            # if batch_norm:  # check also after ReLU
            #    layers.append(nn.BatchNorm1d(channels))

            layers.append(ACTIVATION)

            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))

            self.blocks = nn.Sequential(*layers)

        def forward(self, x):
            output = self.blocks(x)
            return output

    def __init__(self, channels, dropout_rate, skip):
        super().__init__(
            in_channels=channels, out_channels=channels, dropout_rate=dropout_rate,
        )
        self.channels = channels
        self.skip = skip

        self.blocks = nn.Sequential(
            self.InternalResidualBlock(
                channels,
                # batch_norm=batch_norm,
                dropout_rate=dropout_rate,
            ),
            self.InternalResidualBlock(
                channels,
                # batch_norm=batch_norm,
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
                out_channels=td.LATENT_DIM,
                bias=True,
                # batch_norm=config.BATCH_NORM,
                activation=ACTIVATION,
                dropout_rate=False,
            )
        )

        for _ in range(layers_count):
            layers.append(
                ResidualBlock(
                    td.LATENT_DIM,
                    # batch_norm=config.BATCH_NORM,
                    dropout_rate=td.DROPOUT_RATE,
                    skip=td.SKIP,
                )
            )

        layers.append(
            BasicBlock(
                in_channels=td.LATENT_DIM,
                out_channels=output_linear_dim,
                bias=True,  # TODO #65
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


class Attention(Block):
    def __init__(self, in_channels, heads, td: TrainingData):
        super().__init__(
            in_channels=in_channels, out_channels=1, dropout_rate=False,
        )
        self.heads = heads

        if self.heads is None:
            self.blocks = None
            return

        attention_heads = BasicBlock(
            in_channels=td.LATENT_DIM,
            out_channels=self.heads,
            bias=True,
            activation=ACTIVATION,
            dropout_rate=False,
        )

        if self.heads == 1:
            self.blocks = attention_heads
        else:
            self.blocks = nn.Sequential(
                attention_heads, nn.Linear(self.heads, 1, bias=False)
            )

    def forward(self, edge_latents, index):
        if self.blocks is None:
            return 1.0

        alpha_score = self.blocks(edge_latents)
        alpha = softmax(alpha_score, index)
        return alpha


class ProcessorLayer(MessagePassing):
    def __init__(self, td: TrainingData):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=td.LATENT_DIM * 3,
            layers_count=td.PROC_LAYER_COUNT,
            output_linear_dim=td.LATENT_DIM,
            statistics=None,
            batch_norm=td.INTERNAL_BATCH_NORM,
            layer_norm=td.LAYER_NORM,
            td=td,
        )
        self.node_processor = ForwardNet(
            input_dim=td.LATENT_DIM * 2,
            layers_count=td.PROC_LAYER_COUNT,
            output_linear_dim=td.LATENT_DIM,
            statistics=None,
            batch_norm=td.INTERNAL_BATCH_NORM,
            layer_norm=td.LAYER_NORM,
            td=td,
        )

        self.attention = Attention(td.LATENT_DIM, td.ATTENTION_HEADS, td)
        self.epsilon = Parameter(torch.Tensor(1))

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
        aggregated_new_edge_latents = scatter_sum(weighted_edge_latents, index, dim=0)
        return aggregated_new_edge_latents

    def update(self, aggregated_new_edge_latents, node_latents):
        # node_inputs = aggregated_new_edge_latents
        # node_inputs = (
        #    (1 + self.epsilon) * node_latents
        # ) + aggregated_new_edge_latents
        node_inputs = torch.hstack((node_latents, aggregated_new_edge_latents))
        new_node_latents = node_latents + self.node_processor(node_inputs)
        return new_node_latents, self.new_edge_latents


class CustomGraphNet(nn.Module):
    def __init__(
            self, output_dim, statistics: Optional[DatasetStatistics], td: TrainingData,
    ):
        super().__init__()
        self.td = td

        self.node_encoder = ForwardNet(
            input_dim=SettingInput.nodes_data_dim(),
            layers_count=td.ENC_LAYER_COUNT,
            output_linear_dim=td.LATENT_DIM,
            statistics=None if statistics is None else statistics.nodes_statistics,
            batch_norm=td.INPUT_BATCH_NORM,
            layer_norm=td.LAYER_NORM,
            td=td,
        )

        self.edge_encoder = ForwardNet(
            input_dim=SettingInput.edges_data_dim(),
            layers_count=td.ENC_LAYER_COUNT,
            output_linear_dim=td.LATENT_DIM,
            statistics=None if statistics is None else statistics.edges_statistics,
            batch_norm=td.INPUT_BATCH_NORM,
            layer_norm=td.LAYER_NORM,
            td=td,
        )

        self.processor_layers = nn.ModuleList(
            [ProcessorLayer(td) for _ in range(td.MESSAGE_PASSES)]
        )

        self.decoder = ForwardNet(
            input_dim=td.LATENT_DIM,
            layers_count=td.DEC_LAYER_COUNT,
            output_linear_dim=output_dim,
            statistics=None,
            batch_norm=td.INTERNAL_BATCH_NORM,
            layer_norm=False,  # TODO #65
            td=td,
        )

    @property
    def device(self):
        return next(self.parameters()).device

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
            node_latents, edge_latents = processor_layer(
                batch, node_latents, edge_latents
            )

        output = self.decoder(node_latents)
        return output

    # TODO #66

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    # TODO #66

    def solve_all(self, setting):
        self.eval()

        batch = setting.get_data().to(self.device)
        normalized_a_cuda = self(batch)

        normalized_a = thh.to_np_double(normalized_a_cuda)
        a = setting.denormalize_rotate(normalized_a)
        return a, normalized_a

    def solve(self, setting, initial_a):
        a, _ = self.solve_all(setting)
        return a
