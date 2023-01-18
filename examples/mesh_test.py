import pygmsh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from conmech.mesh.boundaries_factory import get_boundary_surfaces


def get_nodes_and_elements(geom, dimension: int):
    geom_mesh = geom.generate_mesh()
    points = geom_mesh.points.copy()
    nodes = points[:, :dimension]
    elements = geom_mesh.cells[-2].data.astype("long").copy()
    return nodes, elements


def main():
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.6],
                [0.0, 1.0, 0.6],
            ]
        )
        geom.extrude(poly, [0.0, 0.0, 0.2], num_layers=3)

        geom.set_mesh_size_callback(lambda dim, tag, x, y, z, *_: 1.0 / 8.0)
        nodes, elements = get_nodes_and_elements(geom, dimension=3)
        print(len(nodes))

        boundary_surfaces, boundary_internal_indices, boundary_indices = get_boundary_surfaces(
            elements
        )
        boundary_surface_nodes = nodes[boundary_surfaces]
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(
                boundary_surface_nodes,
                # edgecolors="tab:orange",
                # linewidths=0.1,
                # facecolors="tab:blue",
                # alpha=0.2,
            )
        )
    plt.savefig("test_mesh.png")


main()
