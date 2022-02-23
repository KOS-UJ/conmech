"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import matplotlib.pyplot as plt
import networkx as nx


class Drawer:

    def __init__(self, state):
        self.state = state
        self.mesh = state.mesh
        self.node_size = 20 + (1000 / len(self.mesh.initial_points))

    def draw(self):
        f, ax = plt.subplots()

        self.draw_mesh(self.mesh.initial_points, ax, label='Original',
                       node_color='0.6', edge_color='0.8')
        self.draw_mesh(self.state.displaced_points, ax, label='Deformed')
        for contact_boundary in self.mesh.boundaries.contact:
            self.draw_boundary(self.state.displaced_points[contact_boundary], ax,
                               edge_color="b")
        for dirichlet_boundary in self.mesh.boundaries.dirichlet:
            self.draw_boundary(self.state.displaced_points[dirichlet_boundary], ax,
                               edge_color="r")
        for dirichlet_boundary in self.mesh.boundaries.neumann:
            self.draw_boundary(self.state.displaced_points[dirichlet_boundary], ax,
                               edge_color="g")

        # turns on axis, since networkx turn them off
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=[handles[0], handles[2]], bbox_to_anchor=(.91, 0.96),
                   bbox_transform=plt.gcf().transFigure)

        f.set_size_inches(10, 4)

        plt.show()

    def draw_mesh(self, vertices, ax, label="", node_color='k', edge_color='k'):
        graph = nx.Graph()
        for i, j, k in self.mesh.cells:
            graph.add_edge(i, j)
            graph.add_edge(i, k)
            graph.add_edge(j, k)

        nx.draw(graph, pos=vertices, label=label, node_color=node_color,
                edge_color=edge_color, node_size=self.node_size, ax=ax)

    def draw_boundary(self, vertices, ax, label="", node_color='k', edge_color='k'):
        graph = nx.Graph()
        for i in range(1, len(vertices)):
            graph.add_edge(i - 1, i)

        nx.draw(graph, pos=vertices, label=label, node_color=node_color,
                edge_color=edge_color, node_size=self.node_size, ax=ax)