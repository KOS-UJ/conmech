import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum
from deep_conmech.common import config, basic_helpers

#TODO: move
ACTIVATION = nn.ReLU()  # nn.PReLU()  # ReLU
#| ac {.ACTIVATION._get_name()} \


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

            self.blocks = nn.Sequential(
                nn.Linear(channels, channels),
                # if batch_norm:  # check also after ReLU
                #    layers.append(nn.BatchNorm1d(channels))
                ACTIVATION,
                nn.Dropout(dropout_rate),
            )

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


class ForwardNet(nn.Module):
    def __init__(
        self, input_dim, layers_count, output_linear_dim, input_normalization=False,
    ):
        super().__init__()

        layers = []
        if input_normalization:
            layers.append(nn.BatchNorm1d(input_dim))

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

        self.net = basic_helpers.set_precision(nn.Sequential(*layers))

    def forward(self, x):
        result = self.net(x)
        return result


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        layers_count,
        output_linear_dim=config.LATENT_DIM,
        input_normalization=False,
        output_bias=True,
    ):
        super().__init__()

        layers = []
        if input_normalization:
            layers.append(nn.BatchNorm1d(input_dim))

        in_channels = input_dim
        for _ in range(layers_count):
            layers.append(
                BasicBlock(
                    in_channels=in_channels,
                    out_channels=config.LATENT_DIM,
                    bias=True,
                    activation=ACTIVATION,
                    dropout_rate=DROPOUT_RATE,
                )
            )
            in_channels = layers[-1].out_channels

        layers.append(
            BasicBlock(
                in_channels=config.LATENT_DIM,
                out_channels=output_linear_dim,
                bias=output_bias,
                activation=ACTIVATION,  ##########################False,
                dropout_rate=False,
            )
        )

        self.net = nn.Sequential(*layers)

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


class ProcessorLayer(MessagePassing):
    def __init__(self):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=config.LATENT_DIM * 3,
            layers_count=config.PROC_LAYER_COUNT,
            output_linear_dim=config.LATENT_DIM,
        )
        self.vertex_processor = ForwardNet(
            input_dim=config.LATENT_DIM * 2,
            layers_count=config.PROC_LAYER_COUNT,
            output_linear_dim=config.LATENT_DIM,
        )

        # self.edge_processor = MLP(input_dim=config.LATENT_DIM * 3)
        # self.vertex_processor = MLP(input_dim=config.LATENT_DIM)  # 2 1
        self.layer_norm = basic_helpers.set_precision(nn.LayerNorm(config.LATENT_DIM))
        self.attention = Attention(config.LATENT_DIM, config.ATTENTION_HEADS)
        self.epsilon = Parameter(torch.Tensor(1))

        # change heads to a
        # self.bias = Parameter(torch.Tensor(out_channels))
        # self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

    def forward(self, batch, vertex_latents, edge_latents):
        self.new_edge_latents = None
        return self.propagate(
            edge_index=batch.edge_index,
            vertex_latents=vertex_latents,
            edge_latents=edge_latents,
        )

    def message(self, vertex_latents_i, vertex_latents_j, edge_latents, index):
        edge_inputs = torch.hstack((vertex_latents_i, vertex_latents_j, edge_latents))
        self.new_edge_latents = edge_latents + self.layer_norm(
            self.edge_processor(edge_inputs)
        )

        alpha = self.attention(edge_latents, index)
        weighted_edge_latents = alpha * self.new_edge_latents
        return weighted_edge_latents

    def aggregate(self, weighted_edge_latents, index):
        aggregated_new_edge_latents = scatter_sum(weighted_edge_latents, index, dim=0)
        return aggregated_new_edge_latents

    def update(self, aggregated_new_edge_latents, vertex_latents):
        # vertex_inputs = aggregated_new_edge_latents
        # vertex_inputs = (
        #    (1 + self.epsilon) * vertex_latents
        # ) + aggregated_new_edge_latents
        vertex_inputs = torch.hstack((vertex_latents, aggregated_new_edge_latents))
        new_vertex_latents = vertex_latents + self.layer_norm(
            self.vertex_processor(vertex_inputs)
        )
        return new_vertex_latents, self.new_edge_latents


class CustomGraphNet(nn.Module):  # SAMPLE
    def __init__(self):
        super().__init__()

        self.vector_encoder = ForwardNet(
            input_dim=config.VERTEX_DATA_DIM,
            layers_count=config.ENC_LAYER_COUNT,
            input_normalization=True,
            output_linear_dim=config.LATENT_DIM,
        )
        self.edge_encoder = ForwardNet(
            input_dim=config.EDGE_DATA_DIM,
            layers_count=config.ENC_LAYER_COUNT,
            input_normalization=True,
            output_linear_dim=config.LATENT_DIM,
        )
        self.layer_norm = basic_helpers.set_precision(nn.LayerNorm(config.LATENT_DIM))

        self.processor_layers = []
        for _ in range(config.MESSAGE_PASSES):
            processor_layer = ProcessorLayer()
            processor_layer.to(basic_helpers.device)
            self.processor_layers.append(processor_layer)

        self.decoder = ForwardNet(
            input_dim=config.LATENT_DIM,
            layers_count=config.DEC_LAYER_COUNT,
            output_linear_dim=config.DIM,
        )
        """
        self.vector_encoder = MLP(
            input_dim=config.VERTEX_DATA_DIM,
            input_normalization=True,  ########################
        )
        self.edge_encoder = MLP(
            input_dim=config.EDGE_DATA_DIM,
            input_normalization=True,  ########################
        )
        self.decoder = MLP(
            input_dim=config.LATENT_DIM,
            output_linear_dim=config.DIM,
            output_bias=False
        )
        """

    def forward(self, batch):
        vertex_input = batch.x  # position "pos" will not generalize
        edge_input = batch.edge_attr

        vertex_latents = self.layer_norm(self.vector_encoder(vertex_input))
        edge_latents = self.layer_norm(self.edge_encoder(edge_input))

        for processor_layer in self.processor_layers:
            vertex_latents, edge_latents = processor_layer(
                batch, vertex_latents, edge_latents
            )

        output = self.decoder(vertex_latents)
        return output


############################
