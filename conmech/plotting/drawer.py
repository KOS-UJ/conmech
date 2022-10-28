"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import matplotlib.pyplot as plt
import matplotlib.patches
import networkx as nx
import numpy as np

from conmech.helpers import cmh
from conmech.helpers.config import Config


class Drawer:
    def __init__(self, state, config: Config):
        self.state = state
        self.config = config
        self.mesh = state.body.mesh
        self.node_size = 2 + (300 / len(self.mesh.initial_nodes))
        self.line_width = self.node_size / 2
        self.deformed_mesh_color = "k"
        self.original_mesh_color = "0.7"
        self.cmap = plt.cm.plasma

    def get_directory(self):
        return f"./output/{self.config.current_time} - DRAWING"

    def draw(self, temp_max=None, temp_min=None, show=True, save=False, save_format="png"):
        fig, axes = plt.subplots()


        x_min = min(min(self.state.body.mesh.initial_nodes[:, 0]),
                    min(self.state.displaced_nodes[:, 0]))
        x_max = max(max(self.state.body.mesh.initial_nodes[:, 0]),
                    max(self.state.displaced_nodes[:, 0]))
        dx = x_max - x_min
        x_margin = dx * 0.2
        xlim = (x_min - x_margin, x_max + x_margin)
        y_min = min(min(self.state.body.mesh.initial_nodes[:, 1]),
                    min(self.state.displaced_nodes[:, 1]))
        y_max = max(max(self.state.body.mesh.initial_nodes[:, 1]),
                    max(self.state.displaced_nodes[:, 1]))
        dy = y_max - y_min
        y_margin = dy * 0.2
        ylim = (y_min - y_margin, y_max + y_margin)

        axes.fill_between(xlim, [ylim[0], ylim[0]], color='blue',alpha=.25)
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)

        if hasattr(self.state, "temperature"):
            temperature = self.state.temperature[:]
            self.draw_field(temperature, temp_min, temp_max, axes, fig)
        if hasattr(self.state, "electric_potential"):
            electric_potential = self.state.electric_potential[:]
            self.draw_field(electric_potential, temp_min, temp_max, axes, fig)

        if self.original_mesh_color is not None:
            self.draw_mesh(
                self.mesh.initial_nodes,
                axes,
                label="Original",
                node_color=self.original_mesh_color,
                edge_color=self.original_mesh_color,
            )

        nodes = self.state.displaced_nodes
        if self.original_mesh_color is not None:
            self.draw_mesh(
                nodes,
                axes,
                label="Deformed",
                node_color=self.deformed_mesh_color,
                edge_color=self.deformed_mesh_color,
            )
        self.draw_boundary(edges=self.mesh.contact_boundary, nodes=nodes, axes=axes, edge_color="b")
        self.draw_boundary(
            edges=self.mesh.dirichlet_boundary, nodes=nodes, axes=axes, edge_color="r"
        )
        self.draw_boundary(edges=self.mesh.neumann_boundary, nodes=nodes, axes=axes, edge_color="g")

        # turns on axis, since networkx turn them off
        plt.axis("on")
        axes.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        axes.set_aspect('equal', adjustable='box')

        # fig.set_size_inches(self.mesh.mesh_prop.scale_x * 12, self.mesh.mesh_prop.scale_y * 16)

        if show:
            fig.tight_layout()
            plt.show()
        if save:
            self.save_plot(save_format)

    def save_plot(self, format_):
        directory = self.get_directory()
        cmh.create_folders(directory)
        path = f"{directory}/{cmh.get_timestamp(self.config)}.{format_}"
        plt.savefig(
            path,
            transparent=False,
            bbox_inches="tight",
            format=format_,
            pad_inches=0.1,
            dpi=800,
        )
        plt.close()

    def draw_mesh(self, nodes, axes, label="", node_color="k", edge_color="k"):
        graph = nx.Graph()
        for i, j, k in self.mesh.elements:
            graph.add_edge(i, j)
            graph.add_edge(i, k)
            graph.add_edge(j, k)

        nx.draw(
            graph,
            pos=nodes,
            label=label,
            node_color=node_color,
            edge_color=edge_color,
            node_size=self.node_size,
            ax=axes,
        )

    def draw_boundary(self, edges, nodes, axes, label="", node_color="k", edge_color="k"):
        graph = nx.Graph()
        for edge in edges:
            graph.add_edge(edge[0], edge[1])

        nx.draw(
            graph,
            pos=nodes,
            label=label,
            node_color=node_color,
            edge_color=edge_color,
            node_size=self.node_size,
            ax=axes,
            width=self.line_width,
        )

    def draw_field(self, field, v_min, v_max, axes, fig):
        x = self.state.displaced_nodes[:, 0]
        y = self.state.displaced_nodes[:, 1]

        n_layers = 100
        axes.tricontour(
            x,
            y,
            self.mesh.elements,
            field,
            15,
            colors='k',
            linewidths=0.2
        )
        axes.tricontourf(
            x,
            y,
            self.mesh.elements,
            field,
            n_layers,
            cmap=self.cmap,
            vmin=v_min,
            vmax=v_max,
        )

        # cbar_ax = f.add_axes([0.875, 0.15, 0.025, 0.6])
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
        sm.set_array([])
        fig.colorbar(sm, fraction=0.046, pad=0.04)
