import os
import time
from typing import List, Optional, Tuple

import matplotlib

import deep_conmech.common.config as config
import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_common
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature
from matplotlib import animation, cm, collections
from matplotlib.patches import Rectangle
from matplotlib.ticker import LinearLocator


def get_fig():
    return plt.figure(figsize=(5, 2))


def get_axs(fig):
    axs = fig.add_subplot(1, 1, 1, facecolor="none")
    return axs


def set_perspective(scale, axs):
    axs.set_aspect("equal", "box")
    padding = 4
    axs.set_xlim(-padding * scale, 20 * scale)
    axs.set_ylim(-padding * scale, padding * scale)


def plot_animation(
    all_setting_paths: List[str],
    time_skip: float,
    path: str,
    t_scale: Optional[List] = None,
):
    plotter_common.plot_animation(
        all_setting_paths=all_setting_paths,
        time_skip=time_skip,
        path=path,
        get_axs=get_axs,
        plot_frame=plot_frame,
        fig=get_fig(),
        t_scale=t_scale,
    )


def plot_frame(
    setting: SettingRandomized,
    current_time: float,
    axs,
    fig,
    draw_detailed: bool = True,
    base_setting: Optional[SettingRandomized] = None,
    t_scale: Optional[List] = None,
):
    scale = setting.mesh_data.scale_x
    set_perspective(scale, axs)

    if isinstance(setting, SettingTemperature):
        draw_main_temperature(axs=axs, fig=fig, setting=setting, t_scale=t_scale)
    else:
        draw_main_displaced(setting, axs)
    if base_setting is not None:
        draw_base_displaced(base_setting, scale, axs)
    description_offset = np.array([-0.1, -1.1]) * scale

    draw_parameters(current_time, setting, scale, axs)
    # draw_angles(setting, ax)

    position = np.array([-1.8, -2.2]) * scale
    shift = 2.5 * scale
    draw_forces(setting, position, axs)
    if draw_detailed:  # detailed:
        position[0] += shift
        if setting.obstacles is not None:
            draw_obstacle_resistance_normalized(setting, position, axs)
            position[0] += shift
        # draw_boundary_faces_normals(setting, position, ax)
        # position[0] += shift
        # draw_boundary_normals(setting, position, ax)
        # position[0] += shift

        draw_boundary_resistance_normal(setting, position, axs)
        position[0] += shift
        draw_boundary_resistance_tangential(setting, position, axs)
        position[0] += shift
        draw_boundary_v_tangential(setting, position, axs)
        position[0] += shift

        draw_input_u(setting, position, axs)
        position[0] += shift
        draw_input_v(setting, position, axs)
        position[0] += shift
        draw_a(setting, position, axs)

        position[0] += shift
        if isinstance(setting, SettingTemperature):
            plot_temperature(
                axs=axs, setting=setting, position=position, t_scale=t_scale
            )

        # draw_edges_data(setting, position, ax)
        # draw_vertices_data(setting, position, ax)

cmap=plt.cm.plasma  # magma plasma

def plot_temperature(axs, setting: SettingTemperature, position, t_scale):
    add_annotation("TEMP", setting, position, axs)
    points = (setting.normalized_points + position).T
    axs.scatter(
        *points,
        c=setting.t_old,
        vmin=t_scale[0],
        vmax=t_scale[1],
        cmap=cmap,
        s=1,
        marker=".",
        linewidths=0.1,
    )

def draw_main_temperature(axs, fig, setting, t_scale):
    draw_main_obstacles(setting, axs)
    values = axs.tricontourf(
        *(setting.moved_nodes.T),
        setting.elements,
        setting.t_old.reshape(-1),
        cmap=cmap,
        vmin=t_scale[0],
        vmax=t_scale[1],
        antialiased=True,
    )
    fig.clim(t_scale[0],t_scale[1])
    fig.colorbar(values, ax=axs)
    #norm = matplotlib.colors.Normalize(vmin=t_scale[0], vmax=t_scale[1])
    #fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    #         cax=axs)#, orientation='horizontal', label='Some Units')


def draw_obstacles(obstacle_origins, obstacle_normals, position, color, ax):
    obstacles_tangient = np.hstack(
        (-obstacle_normals[:, 1, None], obstacle_normals[:, 0, None])
    )
    for i in range(len(obstacle_origins)):
        bias = obstacle_origins[i] + position
        ax.arrow(
            *bias,
            *obstacle_normals[i],
            color=f"tab:{color}",
            alpha=0.4,
            width=0.00002,
            length_includes_head=True,
            head_width=0.01,
        )
        line = ax.axline(
            bias,
            obstacles_tangient[i] + bias,
            color=f"tab:{color}",
            alpha=0.4,
            linewidth=0.4,
        )
        # ax.fill_between(bias, obstacles_tangient[i] + bias)


def plot_arrows(starts, vectors, ax):
    prepared_starts, prepared_vectors = plotter_common.prepare_for_arrows(
        starts, vectors
    )
    ax.quiver(
        *prepared_starts,
        *prepared_vectors,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.001,
        color="w",
        zorder=2,
    )


def draw_main_obstacles(setting, ax):
    draw_obstacles(
        setting.obstacle_origins, setting.obstacle_normals, [0, 0], "orange", ax
    )


def draw_normalized_obstacles(setting, position, ax):
    draw_obstacles(
        setting.normalized_obstacle_origins,
        setting.normalized_obstacle_normals,
        position,
        "blue",
        ax,
    )


def draw_obstacle_resistance_normalized(setting, position, ax):
    draw_normalized_obstacles(setting, position, ax)

    draw_additional_setting("P", setting, position, ax)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.normalized_boundary_penetration,
        ax,
    )


def draw_boundary_normals(setting, position, ax):
    draw_additional_setting("N", setting, position, ax)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.normalized_boundary_normals,
        ax,
    )


def draw_boundary_v_tangential(setting, position, ax):
    draw_additional_setting("V_TNG", setting, position, ax)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.normalized_boundary_v_tangential,
        ax,
    )


def draw_boundary_resistance_normal(setting, position, ax):
    draw_additional_setting("RES_N", setting, position, ax)
    data = setting.normalized_boundary_normals * setting.resistance_normal / 100
    plot_arrows(
        setting.normalized_boundary_nodes + position, data, ax,
    )


def draw_boundary_resistance_tangential(setting, position, ax):
    draw_additional_setting("RES_T", setting, position, ax)
    data = setting.normalized_boundary_normals * setting.resistance_tangential / 100
    plot_arrows(
        setting.normalized_boundary_nodes + position, data, ax,
    )


def draw_rectangle(ax, position, scale_x, scale_y):
    ax.add_patch(
        Rectangle(
            (position[0], position[1],),
            scale_x,
            scale_y,
            fill=None,
            alpha=1.0,
            color="w",
            lw=0.2,
            zorder=3,
        )
    )


def draw_main_displaced(setting, ax):
    position = np.array([0.0, 0.0])
    draw_displaced(setting, position, "orange", ax)
    # draw_points(setting.moved_reference_points, position, "orange", ax)
    if setting.obstacles is not None:
        draw_obstacles(
            setting.obstacle_origins, setting.obstacle_normals, position, "orange", ax,
        )


def draw_base_displaced(setting, scale, ax):
    position = np.array([0.0, 1.5]) * scale
    draw_displaced(setting, position, "purple", ax)
    if setting.obstacles is not None:
        draw_obstacles(
            setting.obstacle_origins, setting.obstacle_normals, position, "orange", ax,
        )


def draw_displaced(setting, position, color, ax):
    # draw_rectangle(ax, position, setting.mesh_data.scale_x, setting.mesh_data.scale_y)
    draw_triplot(setting.moved_nodes + position, setting, f"tab:{color}", ax)
    # draw_data("P", obstacle_forces, setting, [7.5, -1.5], ax)


def draw_points(points, position, color, ax):
    moved_nodes = points + position
    ax.scatter(moved_nodes[:, 0], moved_nodes[:, 1], s=0.1, c=f"tab:{color}")


def draw_forces(setting, position, ax):
    return draw_data("F", setting.normalized_forces, setting, position, ax)


def draw_input_u(setting, position, ax):
    return draw_data("U", setting.input_u_old, setting, position, ax)


def draw_input_v(setting, position, ax):
    return draw_data("V", setting.input_v_old, setting, position, ax)


def draw_a(setting, position, ax):
    return draw_data(
        "A * ts", setting.normalized_a_old * setting.time_step, setting, position, ax,
    )


def draw_data(annotation, data, setting, position, ax):
    draw_additional_setting(annotation, setting, position, ax)
    plot_arrows(setting.normalized_points + position, data, ax)


def draw_additional_setting(annotation, setting, position, ax):
    draw_triplot(setting.normalized_points + position, setting, "tab:blue", ax)
    add_annotation(annotation, setting, position, ax)


def add_annotation(annotation, setting, position, ax):
    scale = setting.mesh_data.scale_x
    description_offset = np.array([-0.5, -1.1]) * scale
    ax.annotate(annotation, xy=position + description_offset, color="w", fontsize=5)


def draw_parameters(time, setting, scale, ax):
    x_max = ax.get_xlim()[1]
    y_max = ax.get_ylim()[1]
    args = dict(color="w", fontsize=5,)
    ax.annotate(
        f"time: {str(round(time, 1))}",
        xy=(x_max - 3.0 * scale, y_max - 0.5 * scale),
        **args,
    )
    ax.annotate(
        f"nodes: {str(setting.nodes_count)}",
        xy=(x_max - 3.0 * scale, y_max - 1.0 * scale),
        **args,
    )


def draw_triplot(nodes, setting, color, ax):
    boundary_nodes = nodes[setting.boundary_faces]
    ax.add_collection(
        collections.LineCollection(
            boundary_nodes,
            colors=[color for _ in range(boundary_nodes.shape[0])],
            linewidths=0.3,
        )
    )
    triplot(nodes, setting.elements, color, ax)


def triplot(nodes, elements, color, ax):
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color=color, linewidth=0.1)


##############


def draw_edges_data(position, setting, ax):
    draw_data_at_edges(setting, setting.edges_data[:, 2:4], position, ax)


def draw_vertices_data(position, setting, ax):
    draw_data_at_vertices(setting, setting.normalized_u_old, position, ax)


def draw_data_at_edges(setting, features, position, ax):
    draw_triplot(setting.normalized_points + position, setting, "tab:orange", ax)

    centers = np.sum(setting.edges_normalized_points + position, axis=1) / 2.0
    vertices = setting.edges_normalized_points[:, 0] + position
    points = (centers + vertices) / 2.0

    for i in range(len(setting.edges_normalized_points)):
        feature = np.around(features[i], 2)
        # np.set_printoptions(precision=3)
        ax.text(
            points[i, 0] - 0.04,
            points[i, 1],
            str(feature),
            # xy=(0.5, 0.5),
            # xycoords="axes fraction",
            fontsize=0.01,
            rotation=30,
            color="w",
        )


def draw_data_at_vertices(setting, features, position, ax):
    draw_triplot(setting.normalized_points + position, setting, "tab:orange", ax)

    points = setting.normalized_points + position
    for i in range(len(setting.normalized_points)):
        feature = np.around(features[i], 2)
        ax.text(
            points[i, 0] - 0.04,
            points[i, 1],
            str(feature),
            # xy=(0.5, 0.5),
            # xycoords="axes fraction",
            fontsize=0.01,
            rotation=30,
            color="w",
        )


###################


def draw_colors_triangles(mesh, data):
    vertices_number = mesh.elements_points.shape[1]
    centers = np.sum(mesh.elements_points, axis=1) / vertices_number

    colors = np.array([(x - 0.5) ** 2 + (y - 0.5) ** 2 for x, y in centers])
    plt.tripcolor(
        mesh.moved_nodes[:, 0],
        mesh.moved_nodes[:, 1],
        mesh.elements,
        facecolors=colors,
        edgecolors="k",
    )
    plt.gca().set_aspect("equal")

    ts = time.time()
    plt_save(f"draw_colors_triangles {ts}")


###################


def draw_mesh_density(id):
    mesh_density = config.MESH_SIZE_PRINT
    corners = config.VAL_PRINT_CORNERS

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    min = nph.min(corners)
    max = nph.max(corners)
    precision = 0.01
    X = np.arange(min[0], max[0], precision)
    Y = np.arange(min[1], max[1], precision)
    X, Y = np.meshgrid(X, Y)

    base_density = nph.get_base_density(mesh_density, corners)
    corner_data = nph.mesh_corner_data(base_density)
    Z = nph.get_adaptive_mesh_density(X, Y, base_density, corner_data)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    max_z = 0.1  # np.max(Z)
    ax.set_zlim(0.0, max_z)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter("{x:.02f}")

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()
    format = "png"
    plt.savefig(
        f"./meshes/mesh_density_{id}.{format}",
        transparent=False,
        bbox_inches="tight",
        format=format,
        pad_inches=0.1,
        dpi=dpi,
    )
    plt.close()
