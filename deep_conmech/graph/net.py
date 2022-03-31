from typing import Optional

import torch
from deep_conmech.common.training_config import TrainingConfig
from deep_conmech.graph.data.dataset_statistics import (DatasetStatistics,
                                                        FeaturesStatistics)
from deep_conmech.graph.helpers import dch, thh
from deep_conmech.graph.setting.setting_input import SettingInput
from torch import layer_norm, nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

# TODO: move
ACTIVATION = nn.ReLU()  # nn.PReLU()  # ReLU
# | ac {.ACTIVATION._get_name()} \


class CustomModule(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def _apply(self, device):
        return super()._apply(device)


class CustomMessagePassing(MessagePassing):
    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device):
        return super().to(device)


class Block(CustomModule):
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


class DataNorm(CustomModule):
    def __init__(self, in_channels, statistics: FeaturesStatistics):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("x_mean", statistics.data_mean)
        self.register_buffer("x_std", statistics.data_std)
        self.register_buffer("mask", statistics.data_std == 0)

    def to(self, device):
        self.x_mean.to(device)
        self.x_std.to(device)
        return super().to(device)

    def forward(self, x):
        output = (x - self.x_mean) / self.x_std
        output = torch.nan_to_num(output)
        return output


class ForwardNet(CustomModule):
    def __init__(
        self,
        input_dim: int,
        layers_count: int,
        output_linear_dim: int,
        statistics: Optional[FeaturesStatistics],
        batch_norm: bool,
        layer_norm: bool,
        config: TrainingConfig,
    ):
        super().__init__()
        layers = []

        self.statistics = statistics
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
            if statistics is not None:
                raise ArgumentException()
        else:
            if statistics is not None:
                layers.append(DataNorm(in_channels=input_dim, statistics=statistics))

        layers.append(
            BasicBlock(
                in_channels=input_dim,
                out_channels=config.LATENT_DIM,
                bias=True,
                # batch_norm=config.BATCH_NORM,
                activation=ACTIVATION,
                dropout_rate=False,
            )
        )

        for _ in range(layers_count):
            layers.append(
                ResidualBlock(
                    config.LATENT_DIM,
                    # batch_norm=config.BATCH_NORM,
                    dropout_rate=config.DROPOUT_RATE,
                    skip=config.SKIP,
                )
            )

        layers.append(
            BasicBlock(
                in_channels=config.LATENT_DIM,
                out_channels=output_linear_dim,
                bias=True,  ################################################################
                # batch_norm=False,
                activation=False,
                dropout_rate=False,
            )
        )

        self.net = thh.set_precision(nn.Sequential(*layers))

        self.layer_norm = (
            thh.set_precision(nn.LayerNorm(config.LATENT_DIM)) if layer_norm else None
        )

    def forward(self, x):
        result = self.net(x)
        return result if self.layer_norm is None else self.layer_norm(result)


class Attention(Block):
    def __init__(self, in_channels, heads, config: TrainingConfig):
        super().__init__(
            in_channels=in_channels, out_channels=1, dropout_rate=False,
        )
        self.heads = heads

        if self.heads is None:
            self.blocks = None
            return

        attention_heads = BasicBlock(
            in_channels=config.LATENT_DIM,
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


class ProcessorLayer(CustomMessagePassing):
    def __init__(self, config: TrainingConfig):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=config.LATENT_DIM * 3,
            layers_count=config.PROC_LAYER_COUNT,
            output_linear_dim=config.LATENT_DIM,
            statistics=None,
            batch_norm=config.INTERNAL_BATCH_NORM,
            layer_norm=config.LAYER_NORM,
            config=config,
        )
        self.node_processor = ForwardNet(
            input_dim=config.LATENT_DIM * 2,
            layers_count=config.PROC_LAYER_COUNT,
            output_linear_dim=config.LATENT_DIM,
            statistics=None,
            batch_norm=config.INTERNAL_BATCH_NORM,
            layer_norm=config.LAYER_NORM,
            config=config,
        )

        self.attention = Attention(config.LATENT_DIM, config.ATTENTION_HEADS, config)
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


class CustomGraphNet(CustomModule):
    def __init__(
        self,
        output_dim,
        statistics: Optional[DatasetStatistics],
        config: TrainingConfig,
    ):
        super().__init__()
        self.config = config

        self.node_encoder = ForwardNet(
            input_dim=SettingInput.nodes_data_dim(),
            layers_count=config.ENC_LAYER_COUNT,
            output_linear_dim=config.LATENT_DIM,
            statistics=None if statistics is None else statistics.nodes_statistics,
            batch_norm=config.INPUT_BATCH_NORM,
            layer_norm=config.LAYER_NORM,
            config=config,
        )

        self.edge_encoder = ForwardNet(
            input_dim=SettingInput.edges_data_dim(),
            layers_count=config.ENC_LAYER_COUNT,
            output_linear_dim=config.LATENT_DIM,
            statistics=None if statistics is None else statistics.edges_statistics,
            batch_norm=config.INPUT_BATCH_NORM,
            layer_norm=config.LAYER_NORM,
            config=config,
        )

        self.processor_layers = nn.ModuleList(
            [ProcessorLayer(config) for _ in range(config.MESSAGE_PASSES)]
        )
        # self.processor_layers.to(config.DEVICE)

        self.decoder = ForwardNet(
            input_dim=config.LATENT_DIM,
            layers_count=config.DEC_LAYER_COUNT,
            output_linear_dim=output_dim,
            statistics=None,
            batch_norm=config.INTERNAL_BATCH_NORM,
            layer_norm=config.LAYER_NORM,
            config=config,
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

        node_latents = self.node_encoder(node_input)
        edge_latents = self.edge_encoder(edge_input)
        
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

        batch = setting.get_data().to(thh.device(self.config))
        normalized_a_cuda = self(batch)

        normalized_a = thh.to_np_double(normalized_a_cuda)
        a = setting.denormalize_rotate(normalized_a)
        return a, normalized_a

    def solve(self, setting, initial_a):
        a, _ = self.solve_all(setting)
        return a
