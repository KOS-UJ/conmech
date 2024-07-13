from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.tri as tri
import scipy.optimize as opt

from conmech.state.products.intersection_contact_limit_points import \
    IntersectionContactLimitPoints
from conmech.state.state import State


def plot_in_columns(states: List[State], *args, **kwargs):
    fig = plt.figure(layout='compressed', figsize=(8.3, 11.7))
    fig.suptitle(kwargs['title'])
    del kwargs['title']

    in3d = kwargs.get('in3d', False)
    if in3d:
        extra_kwargs = {'projection': '3d'}
    else:
        extra_kwargs = {}
        kwargs['in3d'] = False

    cols = 2
    rows = len(states) // 2
    i = 1
    for r in range(1, len(states)+1):
        ax = fig.add_subplot(rows, cols, i, **extra_kwargs)
        do_plot(fig, states[r-1], *args, ax=ax, elev=15, azim=30, **kwargs)
        i += 1
        # ax = fig.add_subplot(rows, cols, i, projection='3d')
        # do_plot(fig, states[r-1], *args, ax=ax, elev=90, azim=90, **kwargs)
        # i += 1
    plt.show()


def plot_limit_points(
        prod: IntersectionContactLimitPoints,
        color='black', title=None, label=None, finish=True, ylim=(0, 1)):
    buff = np.zeros((2, 1000))
    buff_size = 0
    for time, zeros in prod.data.items():
        for zero in zeros:  # significant performance improvement
            buff[0, buff_size] = time
            buff[1, buff_size] = zero
            buff_size += 1
        if buff_size == 1000:
            plt.scatter(buff[0, :], buff[1, :],
                        s=1, color=color, label=label)
            label = None
            buff_size = 0
    if buff_size > 0:
        plt.scatter(buff[0, :buff_size], buff[1, :buff_size],
                    s=1, color=color, label=label)
    plt.title(title)
    plt.ylim(*ylim)

    if finish:
        plt.show()


def plot(state: State, *args, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    do_plot(state, *args, ax=ax, **kwargs)
    plt.show()


def do_plot(
        fig, state: State, field="displacement",
        vmin=None, vmax=None, zmin=None, zmax=None,
        title=None, ax=None, elev=0, azim=-0,
        in3d=False, x=0.0,
):
    assert state.body.mesh.dimension == 2  # membrane have to be 2D
    X = state.body.mesh.nodes[:, 0]
    Y = state.body.mesh.nodes[:, 1]

    soltri = tri.Triangulation(X, Y, triangles=state.body.mesh.elements)
    interpol = tri.LinearTriInterpolator #if in3d else tri.CubicTriInterpolator
    v = interpol(soltri, getattr(state, field)[:, 0])
    u = interpol(soltri, state.displacement[:, 0])

    # higher resolution
    X = np.linspace(0, 1, 100)
    Y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(X, Y)
    X = X.ravel()
    Y = Y.ravel()
    U = np.zeros_like(X)
    for i in range(len(U)):
        U[i] = u(X[i], Y[i])

    B = np.linspace(0, 1, 100)
    RB = np.linspace(1, 0, 100)
    BX = np.concatenate((B,                np.ones_like(B), RB,               np.zeros_like(RB), np.zeros(1)))
    BY = np.concatenate((np.zeros_like(B), B,               np.ones_like(RB), RB,                np.zeros(1)))
    BU = np.empty_like(BX)
    for i in range(len(BX)):
        BU[i] = u(BX[i], BY[i])

    if in3d:
        ax.view_init(elev=elev, azim=azim)
        plot_surface(fig, ax, X, Y, U, BX, BY, BU, lambda x, y, z: v(x, y), vmin, vmax, zmin, zmax, title)
    else:
        plot_intersection(fig, ax, u, x, ymin=zmin, ymax=zmax)


def plot_surface(fig, ax, X, Y, U, BX, BY, BU, v: Callable, vmin=None, vmax=None, zmin=None, zmax=None, title=None):
    p3dc = ax.plot_trisurf(X, Y, U, alpha=1)
    ax.set_zlim(zmin=zmin, zmax=zmax)

    mappable = map_colors(p3dc, v, 'coolwarm', vmin, vmax)
    plt.title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')

    ax.plot3D(BX, BY, BU, color='blue')

    cbar_ax = fig.add_axes([0.10, 0.957, 0.8, 0.02])
    plt.colorbar(mappable, shrink=0.6, aspect=15, cax=cbar_ax, orientation="horizontal")

    # plt.show()

    # v2_mayavi(True, X, Y, U, np.ones_like(U))


def plot_intersection(fig, ax, u, x, ymin, ymax):
    space = np.linspace(0, 1, 100)
    ax.set_ylim([ymin, ymax])
    ax.plot(space, u(np.ones_like(space) * x, space))


def map_colors(p3dc, func, cmap='viridis', vmin=None, vmax=None):
    """
    Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """

    # reconstruct the triangles from internal data
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


def v2_mayavi(transparency, X, Y, Z, O):
    X = X.reshape(100, 100)
    Y = Y.reshape(100, 100)
    Z = Z.reshape(100, 100)
    O = O.reshape(100, 100)
    from mayavi import mlab
    # mlab.test_contour3d()
    # mlab.show()
    # fig = mlab.figure()

    # ax_ranges = [-2, 2, -2, 2, 0, 8]
    # ax_scale = [1.0, 1.0, 0.4]
    # ax_extent = ax_ranges * np.repeat(ax_scale, 2)

    X = np.linspace(0, 1, 100)
    Y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    surf3 = mlab.surf(X, Y, Z)
    # surf4 = mlab.surf(X, Y, O)

    # surf3.actor.actor.scale = ax_scale
    # surf4.actor.actor.scale = ax_scale
    # mlab.view(60, 74, 17, [-2.5, -4.6, -0.3])
    # mlab.outline(surf3, color=(.7, .7, .7),)# extent=ax_extent)
    # mlab.axes(surf3, color=(.7, .7, .7),)# extent=ax_extent,
              #ranges=ax_ranges,
              #label='x', ylabel='y', zlabel='z')

    # if transparency:
    #     surf3.actor.property.opacity = 0.5
    #     surf4.actor.property.opacity = 0.5
        # fig.scene.renderer.use_depth_peeling = 1

    mlab.show()
