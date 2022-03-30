from typing import Optional

import torch
from deep_conmech.common import training_config
from deep_conmech.graph.data.dataset_statistics import (
    DatasetStatistics,
    FeaturesStatistics,
)
from deep_conmech.graph.helpers import dch, thh
from deep_conmech.graph.setting.setting_input import SettingInput
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

# TODO: move
ACTIVATION = nn.ReLU()  # nn.PReLU()  # ReLU
# | ac {.ACTIVATION._get_name()} \


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


class EmptyNorm(nn.Module):
    def forward(self, x):
        return x


class DataNorm(nn.Module):
    def __init__(self, in_channels, normalization_statistics):
        super().__init__()
        self.in_channels = in_channels
        self.x_mean = normalization_statistics.data_mean.to(dch.DEVICE)
        self.x_std = normalization_statistics.data_std.to(dch.DEVICE)
        self.mask = self.x_std == 0

    def forward(self, x):
        output = (x - self.x_mean) / self.x_std
        output = torch.nan_to_num(output)
        return output


class ForwardNet(nn.Module):
    def __init__(self, input_dim, layers_count, output_linear_dim, statistics=None):
        super().__init__()
        layers = []

        self.statistics = statistics
        if statistics is not None:
            layers.append(DataNorm(input_dim, statistics))
        else:
            layers.append(nn.BatchNorm1d(input_dim))

        layers.append(
            BasicBlock(
                in_channels=input_dim,
                out_channels=training_config.LATENT_DIM,
                bias=True,
                # batch_norm=config.BATCH_NORM,
                activation=ACTIVATION,
                dropout_rate=False,
            )
        )

        for _ in range(layers_count):
            layers.append(
                ResidualBlock(
                    training_config.LATENT_DIM,
                    # batch_norm=config.BATCH_NORM,
                    dropout_rate=training_config.DROPOUT_RATE,
                    skip=training_config.SKIP,
                )
            )

        layers.append(
            BasicBlock(
                in_channels=training_config.LATENT_DIM,
                out_channels=output_linear_dim,
                bias=True,  ################################################################
                # batch_norm=False,
                activation=False,
                dropout_rate=False,
            )
        )

        self.net = thh.set_precision(nn.Sequential(*layers))

    def forward(self, x):
        result = self.net(x)
        return result


class Attention(Block):
    def __init__(self, in_channels, heads):
        super().__init__(
            in_channels=in_channels, out_channels=1, dropout_rate=False,
        )
        self.heads = heads

        if self.heads is None:
            self.blocks = None
            return

        attention_heads = BasicBlock(
            in_channels=training_config.LATENT_DIM,
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
    def __init__(self):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=training_config.LATENT_DIM * 3,
            layers_count=training_config.PROC_LAYER_COUNT,
            output_linear_dim=training_config.LATENT_DIM,
        )
        self.node_processor = ForwardNet(
            input_dim=training_config.LATENT_DIM * 2,
            layers_count=training_config.PROC_LAYER_COUNT,
            output_linear_dim=training_config.LATENT_DIM,
        )

        self.layer_norm = (
            thh.set_precision(nn.LayerNorm(training_config.LATENT_DIM))
            if training_config.LAYER_NORM
            else EmptyNorm()
        )
        self.attention = Attention(
            training_config.LATENT_DIM, training_config.ATTENTION_HEADS
        )
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
        self.new_edge_latents = edge_latents + self.layer_norm(
            self.edge_processor(edge_inputs)
        )

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
        new_node_latents = node_latents + self.layer_norm(
            self.node_processor(node_inputs)
        )
        return new_node_latents, self.new_edge_latents


class CustomGraphNet(nn.Module):
    def __init__(self, output_dim, statistics: Optional[DatasetStatistics]):
        super().__init__()

        self.node_encoder = ForwardNet(
            input_dim=SettingInput.nodes_data_dim(),
            layers_count=training_config.ENC_LAYER_COUNT,
            output_linear_dim=training_config.LATENT_DIM,
            statistics=None if statistics is None else statistics.nodes_statistics,
        )
        self.edge_encoder = ForwardNet(
            input_dim=SettingInput.edges_data_dim(),
            layers_count=training_config.ENC_LAYER_COUNT,
            output_linear_dim=training_config.LATENT_DIM,
            statistics=None if statistics is None else statistics.edges_statistics,
        )
        self.layer_norm = (
            thh.set_precision(nn.LayerNorm(training_config.LATENT_DIM))
            if training_config.LAYER_NORM
            else EmptyNorm()
        )

        self.processor_layers = nn.ModuleList(
            [ProcessorLayer() for _ in range(training_config.MESSAGE_PASSES)]
        )
        self.processor_layers.to(dch.DEVICE)

        self.decoder = ForwardNet(
            input_dim=training_config.LATENT_DIM,
            layers_count=training_config.DEC_LAYER_COUNT,
            output_linear_dim=output_dim,
        )

    @property
    def node_statistics(self):
        return self.node_encoder.statistics

    @property
    def edge_statistics(self):
        return self.edge_encoder.statistics

    def forward(self, batch):
        node_input = batch.x  # position "pos" will not generalize
        edge_input = batch.edge_attr

        node_latents = self.layer_norm(self.node_encoder(node_input))
        edge_latents = self.layer_norm(self.edge_encoder(edge_input))

        for processor_layer in self.processor_layers:
            node_latents, edge_latents = processor_layer(
                batch, node_latents, edge_latents
            )

        output = self.decoder(node_latents)
        return output

    ################

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    ################

    def solve_all(self, setting):
        self.eval()

        batch = setting.get_data().to(dch.DEVICE)
        normalized_a_cuda = self(batch)

        normalized_a = thh.to_np_double(normalized_a_cuda)
        a = setting.denormalize_rotate(normalized_a)
        return a, normalized_a

    def solve(self, setting, initial_a):
        a, _ = self.solve_all(setting)
        return a
