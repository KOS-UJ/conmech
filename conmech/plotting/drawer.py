"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from conmech.helpers import cmh
from conmech.helpers.config import Config


class Drawer:
    def __init__(self, state, config: Config):
        """

        outer_forces_scale: if >0 draw outer forces vectors with length scaled by outer_forces_scale
        """
        self.state = state
        self.config = config
        self.mesh = state.body.mesh
        self.node_size = 2 + (300 / len(self.mesh.initial_nodes))
        self.line_width = self.node_size / 2
        self.deformed_mesh_color = "k"
        self.original_mesh_color = "0.7"
        self.outer_forces_scale = 0
        self.normal_stress_scale = 0
        self.field_name = None
        self.field = None
        self.cmap = plt.cm.plasma
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.xlabel = None
        self.ylabel = None

    def get_directory(self):
        return f"./output/{self.config.current_time} - DRAWING"

    def draw(
        self,
        fig_axes=None,
        field_max=None,
        field_min=None,
        show=True,
        save=False,
        save_format="png",
        title=None,
    ):
        fig, axes = fig_axes or plt.subplots()

        if self.x_min is None:
            self.x_min = min(
                min(self.state.body.mesh.initial_nodes[:, 0]), min(self.state.displaced_nodes[:, 0])
            )
        if self.x_max is None:
            self.x_max = max(
                max(self.state.body.mesh.initial_nodes[:, 0]), max(self.state.displaced_nodes[:, 0])
            )
        dx = self.x_max - self.x_min
        x_margin = dx * 0.2
        xlim = (self.x_min - x_margin, self.x_max + x_margin)
        if self.y_min is None:
            self.y_min = min(
                min(self.state.body.mesh.initial_nodes[:, 1]), min(self.state.displaced_nodes[:, 1])
            )
        if self.y_max is None:
            self.y_max = max(
                max(self.state.body.mesh.initial_nodes[:, 1]), max(self.state.displaced_nodes[:, 1])
            )
        dy = self.y_max - self.y_min
        y_margin = dy * 0.2
        ylim = (self.y_min - y_margin, self.y_max + y_margin)

        axes.fill_between(xlim, [ylim[0], ylim[0]], color="blue", alpha=0.25)
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)

        if self.field_name:
            self.field = getattr(self.state, self.field_name)

        if self.field is not None:
            self.draw_field(self.field, field_min, field_max, axes, fig)

        if self.original_mesh_color is not None:
            self.draw_mesh(
                self.mesh.initial_nodes,
                axes,
                label="Original",
                node_color=self.original_mesh_color,
                edge_color=self.original_mesh_color,
            )

        nodes = self.state.displaced_nodes
        if self.deformed_mesh_color is not None:
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

        if self.outer_forces_scale:
            neumann_nodes = self.state.body.mesh.neumann_boundary
            neumann_nodes = list(set(neumann_nodes.flatten()))
            x = self.state.displaced_nodes[neumann_nodes]
            v = self.state.body.node_outer_forces(self.state.time)[neumann_nodes]
            if any(v[:, 0]) or any(v[:, 1]):  # to avoid warning
                axes.quiver(x[:, 0], x[:, 1], v[:, 0], v[:, 1],
                            angles='xy', scale_units='xy', scale=self.outer_forces_scale)

        if self.normal_stress_scale:
            contact_nodes = self.state.body.mesh.contact_boundary
            contact_nodes = list(set(contact_nodes.flatten()))
            x = self.state.displaced_nodes[contact_nodes]
            v = np.zeros((len(contact_nodes), 2))  # TODO
            v[:, 1] = - self.state.stress_y[contact_nodes]  # TODO
            if any(v[:, 0]) or any(v[:, 1]):  # to avoid warning
                axes.quiver(x[:, 0], x[:, 1], v[:, 0], v[:, 1],
                            angles='xy', scale_units='xy', scale=self.normal_stress_scale)

        # turns on axis, since networkx turn them off
        plt.axis("on")
        axes.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        axes.set_aspect("equal", adjustable="box")
        plt.title(title)

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
        axes.tricontour(x, y, self.mesh.elements, field, 15, colors="k", linewidths=0.2)
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

        # cbar_ax = fig.add_axes([0.875, 0.15, 0.025, 0.6])
        # ax_pos = axes.get_position()
        # cax = fig.add_axes(
        #     [axes.get_position().x0, axes.get_position().y0 * 0, axes.get_position().width, axes.get_position().height * 0.05])

        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("bottom", size="5%", pad=0.15)
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
        sm.set_array([])
        fig.colorbar(sm, orientation="horizontal", label="Norm of stress tensor", ax=axes)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
