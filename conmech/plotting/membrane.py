from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib import tri

from conmech.state.products.intersection_contact_limit_points import (
    VerticalIntersectionContactLimitPoints,
)
from conmech.state.state import State


def plot_in_columns(states: List[State], *args, **kwargs):
    fig = plt.figure(layout="compressed", figsize=(8.3, 11.7))
    fig.suptitle(kwargs["title"])
    del kwargs["title"]

    in3d = kwargs.get("in3d", False)
    if in3d:
        extra_kwargs = {"projection": "3d"}
    else:
        extra_kwargs = {}
        kwargs["in3d"] = False

    cols = 2
    rows = len(states) // 2
    i = 1
    for r in range(1, len(states) + 1):
        ax = fig.add_subplot(rows, cols, i, **extra_kwargs)
        do_plot(fig, states[r - 1], *args, ax=ax, elev=15, azim=30, **kwargs)
        i += 1
        # ax = fig.add_subplot(rows, cols, i, projection='3d')
        # do_plot(fig, states[r-1], *args, ax=ax, elev=90, azim=90, **kwargs)
        # i += 1
    if kwargs["finish"]:
        plt.show()


def plot_limit_points(
    prod: VerticalIntersectionContactLimitPoints,
    color="black",
    title=None,
    label=None,
    finish=True,
    ylim=(0, 1),
):
    buff = np.zeros((2, 1100))  # some extra space for multiple zeros
    buff_size = 0
    for time, zeros in prod.data.items():
        for zero in zeros:  # significant performance improvement
            buff[0, buff_size] = time
            buff[1, buff_size] = zero
            buff_size += 1
        if buff_size == 1000:
            plt.scatter(buff[0, :], buff[1, :], s=1, color=color, label=label)
            label = None
            buff_size = 0
    if buff_size > 0:
        plt.scatter(buff[0, :buff_size], buff[1, :buff_size], s=1, color=color, label=label)
    plt.title(title)
    plt.ylim(*ylim)

    if finish:
        plt.show()


def plot(state: State, *args, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    do_plot(state, *args, ax=ax, **kwargs)
    plt.show()


# pylint: disable=too-many-arguments, too-many-locals
def do_plot(
    fig,
    state: State,
    field="displacement",
    vmin=None,
    vmax=None,
    zmin=None,
    zmax=None,
    title=None,
    ax=None,
    elev=0,
    azim=-0,
    in3d=False,
    x=0.0,
    **_kwargs,
):
    assert state.body.mesh.dimension == 2  # membrane have to be 2D
    mesh_node_x = state.body.mesh.nodes[:, 0]
    mesh_node_y = state.body.mesh.nodes[:, 1]

    soltri = tri.Triangulation(mesh_node_x, mesh_node_y, triangles=state.body.mesh.elements)
    interpol = tri.LinearTriInterpolator  # if in3d else tri.CubicTriInterpolator
    v = interpol(soltri, getattr(state, field)[:, 0])
    u = interpol(soltri, state.displacement[:, 0])

    # higher resolution
    x_disc = np.linspace(0, 1, 100)
    y_disc = np.linspace(0, 1, 100)
    node_x, node_y = np.meshgrid(x_disc, y_disc)
    node_x = node_x.ravel()
    node_y = node_y.ravel()
    node_val = np.zeros_like(node_x)
    # pylint: disable=consider-using-enumerate
    for i in range(len(node_val)):
        node_val[i] = u(node_x[i], node_y[i])

    bound = np.linspace(0, 1, 100)
    rev_bound = np.linspace(1, 0, 100)
    bound_x = np.concatenate(
        (bound, np.ones_like(bound), rev_bound, np.zeros_like(rev_bound), np.zeros(1))
    )
    bound_y = np.concatenate(
        (np.zeros_like(bound), bound, np.ones_like(rev_bound), rev_bound, np.zeros(1))
    )
    bound_val = np.empty_like(bound_x)
    # pylint: disable=consider-using-enumerate
    for i in range(len(bound_x)):
        bound_val[i] = u(bound_x[i], bound_y[i])

    if in3d:
        ax.view_init(elev=elev, azim=azim)
        plot_surface(
            fig,
            ax,
            node_x,
            node_y,
            node_val,
            bound_x,
            bound_y,
            bound_val,
            lambda x, y, z: v(x, y),
            vmin,
            vmax,
            zmin,
            zmax,
            title,
        )
    else:
        plot_intersection(ax, u, x, ymin=zmin, ymax=zmax)


def plot_surface(
    fig,
    ax,
    node_x,
    node_y,
    node_val,
    bound_x,
    bound_y,
    bound_val,
    v_func: Callable,
    vmin=None,
    vmax=None,
    zmin=None,
    zmax=None,
    title=None,
):
    p3dc = ax.plot_trisurf(node_x, node_y, node_val, alpha=1)
    ax.set_zlim(zmin=zmin, zmax=zmax)

    mappable = map_colors(p3dc, v_func, "coolwarm", vmin, vmax)
    plt.title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    ax.plot3D(bound_x, bound_y, bound_val, color="blue")

    cbar_ax = fig.add_axes([0.10, 0.957, 0.8, 0.02])
    plt.colorbar(mappable, shrink=0.6, aspect=15, cax=cbar_ax, orientation="horizontal")

    # plt.show()


def plot_intersection(ax, valfunc, insec_x, ymin, ymax):
    space = np.linspace(0, 1, 100)
    ax.set_ylim([ymin, ymax])
    ax.plot(space, valfunc(np.ones_like(space) * insec_x, space))


def map_colors(p3dc, func, cmap="viridis", vmin=None, vmax=None):
    """
    Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """

    # reconstruct the triangles from internal data
    # pylint: disable=protected-access
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = np.array([np.array((x[s], y[s], z[s])).T for s in slices])

    # compute the barycentres for each triangle
    xb, yb, zb = triangles.mean(axis=1).T

    # compute the function in the barycentres
    values = func(xb, yb, zb)

    # usual stuff
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = get_cmap(cmap)(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)
