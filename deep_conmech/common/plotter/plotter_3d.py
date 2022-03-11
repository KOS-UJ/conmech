# %%
import matplotlib.pyplot as plt
import numpy as np
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.simulator.setting.setting_mesh import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_ax(fig, grid, angle, distance):
    ax = fig.add_subplot(grid, projection="3d", facecolor="none")  # none") #000000
    ax.set_proj_type("ortho")
    ax.view_init(elev=angle[0], azim=angle[1])  # , vertical_axis='y')
    ax.dist = distance

    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    aspect = (12, 4, 4)
    ax.set_box_aspect(aspect)

    ax.set_xlim(-1, aspect[0] - 1)
    ax.set_ylim(-1, aspect[1] - 1)
    ax.set_zlim(-1, aspect[2] - 1)

    ax.set_xlabel("x", labelpad = 0.05, color="w")
    ax.set_ylabel("y", labelpad = 0.05, color="w")
    ax.set_zlabel("z", labelpad = 0.05, color="w")

    ticks = []  # np.arange(0, 2, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.tick_params(axis="x", colors="w")
    ax.tick_params(axis="y", colors="w")
    ax.tick_params(axis="z", colors="w")

    ax.w_xaxis.line.set_color("w")
    ax.w_yaxis.line.set_color("w")
    ax.w_zaxis.line.set_color("w")

    return ax


def plot_frame(setting, normalized_data, path, extension):
    plot_subframe_lambda = lambda ax: plot_subframe(
        setting, normalized_data=normalized_data, ax=ax,
    )

    angles = np.array([[[0, -90], [0, 0]], [[30, -60], [90, 0]]])
    distances = np.array([[10, 10], [11, 10]])
    rows, columns, _ = angles.shape

    fig = plt.figure()  # constrained_layout=True)
    grid = fig.add_gridspec(nrows=rows, ncols=columns)
    # , width_ratios=[1, 1.], height_ratios=[1., 1.])
    # fig.subplots_adjust(left=-0.2, bottom=0., right=1., top=1.)#, wspace=-0.4, hspace=-0.4)

    ax1 = get_ax(fig, grid[0, 0], angles[0, 0], distances[0, 0])
    ax1.set_position([0.6, 0.8, 0.4, 0.4])
    plot_subframe_lambda(ax1)

    ax2 = get_ax(fig, grid[0, 1], angles[0, 1], distances[0, 1])
    ax2.set_position([1.0, 0.8, 0.4, 0.4])
    plot_subframe_lambda(ax2)

    ax3 = get_ax(fig, grid[1, 0], angles[1, 0], distances[1, 0])
    ax3.set_position([0.5, 0.3, 0.7, 0.7])
    plot_subframe_lambda(ax3)

    ax4 = get_ax(fig, grid[1, 1], angles[1, 1], distances[1, 1])
    ax4.set_position([1.0, 0.5, 0.4, 0.4])
    plot_subframe_lambda(ax4)



def draw_base_arrows(ax, base):
    z = np.array([0, 2.0, 2.0])
    ax.quiver(*z, *(base[0]), arrow_length_ratio=0.1, color="r")
    ax.quiver(*z, *(base[1]), arrow_length_ratio=0.1, color="y")
    ax.quiver(*z, *(base[2]), arrow_length_ratio=0.1, color="g")


def plot_arrows(ax, starts, vectors):
    for i in range(len(starts)):
        ax.quiver(*starts[i], *vectors[i], arrow_length_ratio=0.1, color="w", lw=0.2)


def plot_subframe(
    setting, normalized_data, ax,
):
    draw_base_arrows(ax, setting.moved_base)

    """
    moved_boundary_faces_nodes = moved_nodes[boundary_faces]
    moved_boundary_faces_centers = np.mean(moved_boundary_faces_nodes, axis=1)
    plot_arrows(ax, starts=moved_boundary_faces_centers, vectors=boundary_faces_normals)

    moved_boundary_internal_nodes = moved_nodes[boundary_internal_indices]
    ax.scatter(
       moved_boundary_internal_nodes[:, 0], moved_boundary_internal_nodes[:, 1], moved_boundary_internal_nodes[:, 2],
       s=0.05,
       color="w"
    )
    all_boundary_nodes = moved_boundary_faces_nodes.reshape(-1,3)
    ax.scatter(
       all_boundary_nodes[:, 0], all_boundary_nodes[:, 1], all_boundary_nodes[:, 2],
       s=0.05,
       color="w"
    )
    """

    plot_mesh(
        ax, setting.moved_nodes, setting, "tab:orange",
    )
    plot_obstacles(ax, setting, "tab:orange")

    shifted_normalized_nodes = setting.normalized_points + np.array([0, 2.0, 0])
    for data in normalized_data:
        plot_arrows(ax, starts=shifted_normalized_nodes, vectors=data)

        plot_mesh(
            ax, shifted_normalized_nodes, setting, "tab:blue",
        )

        shifted_normalized_nodes = shifted_normalized_nodes + np.array([2.5, 0, 0])


def plot_mesh(ax, nodes, setting, color):
    boundary_faces_nodes = nodes[setting.boundary_faces]
    ax.add_collection3d(
        Poly3DCollection(
            boundary_faces_nodes,
            edgecolors=color,
            linewidths=0.1,
            facecolors=color,
            alpha=0.2,
        )
    )
    """
    for e in elements:
        pts = nodes[e, :]
        for i in range(EDIM):
            for j in range(EDIM):
                if i < j:
                    ax.plot3D(
                        pts[[i, j], 0], pts[[i, j], 1], pts[[i, j], 2], color=color, lw=0.4
                    )

    boundary_nodes = nodes[nodes_indices]
    ax.scatter(
       boundary_nodes[:, 0], boundary_nodes[:, 1], boundary_nodes[:, 2], color="b"
    )
    """


def plot_obstacles(ax, setting, color):
    alpha = 0.3
    node = setting.obstacle_nodes[0]
    normal = setting.obstacle_normals[0]

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -node.dot(normal)

    x_rng = np.arange(-1.2, 11.2, 0.2)
    y_rng = np.arange(-1.2, 3.2, 0.2)
    X, Y = np.meshgrid(x_rng, y_rng)
    Z = (-normal[0] * X - normal[1] * Y - d) / normal[2]
    col = (Z[0,:] > -1.2) & (Z[0,:] < 3.2)

    ax.plot_surface(X[:,col], Y[:,col], Z[:,col], color=color, alpha=alpha)

    ax.quiver(
        *node,
        *normal,
        color=color,
        alpha=alpha
    )


def plt_save(path, extension):
    plt.savefig(
        path,
        transparent=False,
        facecolor="#24292E",  # AAAAAA',
        bbox_inches="tight",
        pad_inches=0.0,
        format=extension,
        dpi=1000,  # 800 1200,
    )
    plt.close()
