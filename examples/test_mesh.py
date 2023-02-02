import numba
import numpy as np
import pygmsh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from conmech.helpers.tmh import Timer
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


# main()


def find_instr(func, keyword, sig=0, limit=5):
    print("test")
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split("\n"):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print("No instructions found")
    print(count)


@numba.njit()
def iterate_test_loop(values):
    c = 0
    mask = np.empty_like(values)
    for i in range(len(values)):
        for j in range(len(values)):
            x = values[j] - values[i]
            c += 1
    return c


@numba.njit()
def iterate_test_broadcast(values):
    c = 0
    for i in range(len(values)):
        mask = values - values[i]
        for j in range(len(values)):
            a = mask[i]
            c += 1
    return c


values = np.zeros((10000, 3))

timer = Timer()

with timer["iterate_test_broadcast"]:
    result_broadcast = iterate_test_broadcast(values)

with timer["iterate_test_loop"]:
    result_loop = iterate_test_loop(values)

with timer["iterate_test_loop2"]:
    result_loop = iterate_test_loop(values)

print(timer.to_dataframe())
for v, k in iterate_test_broadcast.inspect_llvm().items():
    print(v, k)
# for v, k in iterate_test_loop.inspect_llvm().items():
#     print(v, k)

x = 0
