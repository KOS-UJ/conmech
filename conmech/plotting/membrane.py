from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.tri as tri

from conmech.state.state import State


def plot(state: State, field="displacement", vmin=None, vmax=None, zmin=None, zmax=None, title=None):
    assert state.body.mesh.dimension == 2  # membrane have to be 2D
    X = state.body.mesh.nodes[:, 0]
    Y = state.body.mesh.nodes[:, 1]
    U = state.displacement[:, 0]

    soltri = tri.Triangulation(X, Y, triangles=state.body.mesh.elements)
    v = tri.LinearTriInterpolator(soltri, getattr(state, field)[:, 0])

    plot_surface(X, Y, U, lambda x, y, z: v(x, y), vmin, vmax, zmin, zmax, title)


def plot_surface(X, Y, U, v: Callable, vmin=None, vmax=None, zmin=None, zmax=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    triang = mtri.Triangulation(X, Y)

    p3dc = ax.plot_trisurf(triang, U)
    ax.set_zlim(zmin=zmin, zmax=zmax)

    mappable = map_colors(p3dc, v, 'coolwarm', vmin, vmax)
    plt.colorbar(mappable, shrink=0.67, aspect=16.7)
    plt.title(title)

    ax.view_init(elev=0, azim=-45)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()


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
