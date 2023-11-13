"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize

from conmech.helpers.config import Config


class Drawer:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, state, config: Config):
        """
        outer_forces_scale: if >0 draw outer forces vectors with length scaled
                            (divided) by outer_forces_scale
                            if <0 draw fixed length vectors (e.g. -1 means vectors equals 1)
        """
        self.state = state
        self.config = config
        self.mesh = state.body.mesh
        self.node_size = 2 + (300 / len(self.mesh.nodes))
        self.line_width = self.node_size / 2
        self.deformed_mesh_color = "k"
        self.original_mesh_color = "0.7"
        self.outer_forces_scale = 0
        self.normal_stress_scale = 0
        self.field_name = None
        self.field_label = None
        self.field = None
        self.colorful = True
        self.cmap = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.xlabel = None
        self.ylabel = None

    # pylint: disable=too-many-arguments
    def draw(
        self,
        fig_axes=None,
        field_max=None,
        field_min=None,
        show=True,
        save=False,
        save_format="png",
        title=None,
        foundation=True,
    ):
        fig, axes = fig_axes or plt.subplots()

        self.set_axes_limits(axes, foundation)

        if self.field_name:
            self.field = getattr(self.state, self.field_name)
            if self.field_label is None:
                self.field_label = self.field_name
        if self.field is not None:
            self.draw_field(self.field, field_min, field_max, axes, fig)

        self.draw_meshes(axes)

        self.draw_boundaries(axes)

        self.draw_forces(axes)
        self.draw_stress(axes)

        # turns on axis, since networkx turn them off
        # plt.axis("on")
        # axes.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        #
        axes.set_aspect("equal", adjustable="box")
        if title is not None:
            plt.title(title)

        if show:
            fig.tight_layout()
            plt.show()
        if save:
            self.save_plot(save_format, name=save)

    def set_axes_limits(self, axes, foundation):
        # pylint: disable=nested-min-max
        if self.x_min is None:
            self.x_min = min(
                min(self.state.body.mesh.nodes[:, 0]),
                min(self.state.displaced_nodes[:, 0]),
            )
        if self.x_max is None:
            self.x_max = max(
                max(self.state.body.mesh.nodes[:, 0]),
                max(self.state.displaced_nodes[:, 0]),
            )
        dx = self.x_max - self.x_min
        x_margin = dx * 0.2
        xlim = (self.x_min - x_margin, self.x_max + x_margin)
        if self.y_min is None:
            self.y_min = min(
                min(self.state.body.mesh.nodes[:, 1]),
                min(self.state.displaced_nodes[:, 1]),
            )
        if self.y_max is None:
            self.y_max = max(
                max(self.state.body.mesh.nodes[:, 1]),
                max(self.state.displaced_nodes[:, 1]),
            )
        dy = self.y_max - self.y_min
        y_margin = dy * 0.2
        ylim = (self.y_min - y_margin, self.y_max + y_margin)

        if foundation:
            axes.fill_between(xlim, [ylim[0], ylim[0]], color="gray", alpha=0.25)
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)
        if self.xlabel is not None:
            axes.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            axes.set_ylabel(self.ylabel)

    def draw_meshes(self, axes):
        if self.original_mesh_color is not None:
            self.draw_mesh(
                self.mesh.nodes,
                axes,
                label="Original",
                node_color=self.original_mesh_color,
                edge_color=self.original_mesh_color,
            )

        if self.deformed_mesh_color is not None:
            self.draw_mesh(
                self.state.displaced_nodes,
                axes,
                label="Deformed",
                node_color=self.deformed_mesh_color,
                edge_color=self.deformed_mesh_color,
            )

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

    def draw_boundaries(self, axes):
        if self.colorful:
            nodes = self.state.displaced_nodes
            self.draw_boundary(
                edges=self.mesh.contact_boundary, nodes=nodes, axes=axes, edge_color="b"
            )
            self.draw_boundary(
                edges=self.mesh.dirichlet_boundary,
                nodes=nodes,
                axes=axes,
                edge_color="r",
            )
            self.draw_boundary(
                edges=self.mesh.neumann_boundary, nodes=nodes, axes=axes, edge_color="g"
            )
        else:
            self.draw_dirichlet(axes)

    def draw_dirichlet(self, axes):
        dirichlet_nodes = self.state.body.mesh.dirichlet_boundary
        dirichlet_nodes = list(set(dirichlet_nodes.flatten()))
        x = self.state.displaced_nodes[dirichlet_nodes][[0, 1]]
        v = np.zeros_like(x) + np.asarray([1, 0])
        if any(v[:, 0]) or any(v[:, 1]):  # to avoid warning
            axes.quiver(
                x[:, 0],
                x[:, 1],
                v[:, 0],
                v[:, 1],
                angles="xy",
                scale_units="xy",
                scale=8.0,
                headlength=10,
                headaxislength=10,
                headwidth=10,
                pivot="tip",
                edgecolor="k",
                facecolor="None",
                linewidth=0.5,
            )

    def draw_forces(self, axes):
        if self.outer_forces_scale:
            neumann_nodes = self.state.body.mesh.neumann_boundary
            neumann_nodes = list(set(neumann_nodes.flatten()))
            x = self.state.displaced_nodes[neumann_nodes]
            v = self.state.body.dynamics.force.outer.node_source(
                self.state.body.mesh.nodes, self.state.time
            )[neumann_nodes]

            scale = self.outer_forces_scale

            if self.outer_forces_scale < 0:
                v = normalize(v)
                v *= -self.outer_forces_scale
                scale = 1

            pivot = "tip" if any(v[:, 1] < 0) else "tail"  # TODO
            if any(v[:, 0]) or any(v[:, 1]):  # to avoid warning
                axes.quiver(
                    x[:, 0],
                    x[:, 1],
                    v[:, 0],
                    v[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=scale,
                    pivot=pivot,
                )

    def draw_stress(self, axes):
        if self.normal_stress_scale:
            contact_nodes = self.state.body.mesh.contact_boundary
            contact_nodes = list(set(contact_nodes.flatten()))
            x = self.state.displaced_nodes[contact_nodes]
            v = np.zeros((len(contact_nodes), 2))  # TODO
            v[:, 1] = -self.state.stress_y[contact_nodes]  # TODO
            if any(v[:, 0]) or any(v[:, 1]):  # to avoid warning
                axes.quiver(
                    x[:, 0],
                    x[:, 1],
                    v[:, 0],
                    v[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=self.normal_stress_scale,
                )

    @staticmethod
    def get_output_path(config, format_, name):
        output_dir = config.output_dir or str(config.timestamp)
        directory = f"{config.outputs_path}/{output_dir}"
        name = name if isinstance(name, str) else time.time_ns()
        path = f"{directory}/{name}.{format_}"
        return path

    def save_plot(self, format_, name=None):
        path = self.get_output_path(self.config, format_, name)
        plt.savefig(
            path,
            transparent=False,
            bbox_inches="tight",
            format=format_,
            pad_inches=0.1,
            dpi=800,
        )
        plt.close()

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
        #     [axes.get_position().x0, axes.get_position().y0 * 0,
        #     axes.get_position().width, axes.get_position().height * 0.05])

        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("bottom", size="5%", pad=0.15)
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
        sm.set_array([])
        fig.colorbar(sm, orientation="horizontal", label=self.field_label, ax=axes)
