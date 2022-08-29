from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections
from matplotlib.patches import Rectangle

from conmech.helpers.config import Config
from conmech.plotting import plotter_common
from conmech.plotting.plotter_common import PlotAnimationConfig, make_animation
from conmech.scene.scene import Scene
from conmech.scene.scene_temperature import SceneTemperature


def get_fig():
    return plt.figure(figsize=(4, 2))


def get_axs(fig):
    # axs = fig.add_subplot(1, 1, 1, facecolor="none")
    axs = fig.add_axes([0.075, 0.15, 0.9, 0.8], facecolor="none")
    return axs


def set_perspective(scale, axes):
    axes.set_aspect("equal", "box")
    padding = 6
    axes.set_xlim(-padding * scale, 18 * scale)
    axes.set_ylim(-padding * scale, padding * scale)
    plotter_common.set_ax(axes)


def plot_animation(
    save_path: str,
    config: Config,
    time_skip: float,
    index_skip: int,
    plot_scenes_count: int,
    all_scenes_path: str,
    all_calc_scenes_path: Optional[str],
    t_scale: Optional[np.ndarray] = None,
):
    animate = make_animation(get_axs, plot_frame, t_scale)
    plotter_common.plot_animation(
        animate=animate,
        fig=get_fig(),
        config=config,
        plot_config=PlotAnimationConfig(
            save_path=save_path,
            time_skip=time_skip,
            index_skip=index_skip,
            plot_scenes_count=plot_scenes_count,
            all_scenes_path=all_scenes_path,
            all_calc_scenes_path=all_calc_scenes_path,
        ),
    )


def plot_frame(
    fig,
    axs,
    scene: Scene,
    current_time: float,
    draw_detailed: bool = True,
    base_scene: Optional[Scene] = None,
    t_scale: Optional[np.ndarray] = None,
):
    axes = axs
    scale = scene.mesh_prop.scale_x
    set_perspective(scale, axes=axes)

    if isinstance(scene, SceneTemperature):
        cbar_settings = plotter_common.get_t_data(t_scale)
        plotter_common.plot_colorbar(fig, axs=[axes], cbar_settings=cbar_settings)
        draw_main_temperature(axes=axes, scene=scene, cbar_settings=cbar_settings)
    else:
        draw_main_displaced(scene, axes=axes)
    if base_scene is not None:
        draw_base_displaced(base_scene, axes=axes)

    draw_parameters(current_time, scene, scale, axes=axes)
    # draw_angles(scene, axes)

    if draw_detailed:
        position = np.array([-3.7, 4.2]) * scale
        draw_all_sparse(scene, position, axes=axes)

        position = np.array([-6.2, -4.2]) * scale
        shift = 2.5 * scale
        position[0] += shift
        draw_initial(scene, position, axes=axes)
        position[0] += shift
        draw_forces(scene, position, axes=axes)

        position[0] += shift
        draw_obstacle_resistance_normalized(scene, position, axes=axes)
        position[0] += shift
        # draw_boundary_surfaces_normals(scene, position, axes)
        # position[0] += shift
        # draw_boundary_normals(scene, position, axes)
        # position[0] += shift

        draw_boundary_resistance_normal(scene, position, axes=axes)
        position[0] += shift
        draw_boundary_resistance_tangential(scene, position, axes=axes)
        position[0] += shift
        draw_boundary_v_tangential(scene, position, axes=axes)
        position[0] += shift

        draw_input_u(scene, position, axes=axes)
        position[0] += shift
        draw_input_v(scene, position, axes=axes)
        position[0] += shift
        draw_a(scene, position, axes=axes)

        position[0] += shift
        if isinstance(scene, SceneTemperature):
            plot_temperature(
                axes=axes,
                scene=scene,
                position=position,
                cbar_settings=cbar_settings,
            )

        # draw_edges_data(scene, position, axes)
        # draw_vertices_data(scene, position, axes)


def plot_temperature(
    axes,
    scene: SceneTemperature,
    position,
    cbar_settings: plotter_common.ColorbarSettings,
):
    add_annotation("TEMP", scene, position, axes)
    nodes = (scene.normalized_nodes + position).T
    axes.scatter(
        *nodes,
        c=scene.t_old,
        cmap=cbar_settings.cmap,
        vmin=cbar_settings.vmin,
        vmax=cbar_settings.vmax,
        s=1,
        marker=".",
        linewidths=0.1,
    )


def draw_main_temperature(axes, scene, cbar_settings):
    draw_main_obstacles(scene, axes)
    axes.tricontourf(
        *scene.moved_nodes.T,
        scene.elements,
        scene.t_old.reshape(-1),
        cmap=cbar_settings.cmap,
        vmin=cbar_settings.vmin,
        vmax=cbar_settings.vmax,
        antialiased=True,
    )


def draw_obstacles(obstacle_nodes, obstacle_normals, position, color, axes):
    if len(obstacle_nodes) == 0:
        return

    obstacles_tangient = np.hstack((-obstacle_normals[:, 1, None], obstacle_normals[:, 0, None]))
    for i, obstacle_node in enumerate(obstacle_nodes):
        bias = obstacle_node + position
        axes.arrow(
            *bias,
            *obstacle_normals[i],
            color=f"tab:{color}",
            alpha=0.4,
            width=0.00002,
            length_includes_head=True,
            head_width=0.01,
        )
        axes.axline(
            bias,
            obstacles_tangient[i] + bias,
            color=f"tab:{color}",
            alpha=0.4,
            linewidth=0.4,
        )


def plot_arrows(starts, vectors, axes):
    prepared_starts, prepared_vectors = plotter_common.prepare_for_arrows(starts, vectors)
    axes.quiver(
        *prepared_starts,
        *prepared_vectors,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.001,
        color="w",
        zorder=2,
    )


def draw_initial(scene: Scene, position, axes):
    draw_initial_body("I", scene, position, axes)


def draw_main_obstacles(scene: Scene, axes):
    draw_obstacles(
        scene.linear_obstacle_nodes, scene.linear_obstacle_normals, [0, 0], "orange", axes
    )


def draw_normalized_obstacles(scene: Scene, position, axes):
    draw_obstacles(
        scene.normalized_obstacle_nodes,
        scene.get_norm_obstacle_normals(),
        position,
        "blue",
        axes,
    )


def draw_obstacle_resistance_normalized(scene: Scene, position, axes):
    draw_moved_body("P", scene, position, axes)
    plot_arrows(
        scene.normalized_boundary_nodes + position,
        scene.get_normalized_boundary_penetration(),
        axes,
    )


def draw_boundary_normals(scene: Scene, position, axes):
    draw_moved_body("N", scene, position, axes)
    plot_arrows(
        scene.normalized_boundary_nodes + position,
        scene.get_normalized_boundary_normals(),
        axes,
    )


def draw_boundary_v_tangential(scene: Scene, position, axes):
    draw_moved_body("V_TNG", scene, position, axes)
    plot_arrows(
        scene.normalized_boundary_nodes + position,
        scene.get_friction_input(),
        axes,
    )


def draw_boundary_resistance_normal(scene: Scene, position, axes):
    draw_moved_body("RES_N", scene, position, axes)
    data = scene.get_normalized_boundary_normals() * scene.get_resistance_normal() / 100
    plot_arrows(
        scene.normalized_boundary_nodes + position,
        data,
        axes,
    )


def draw_boundary_resistance_tangential(scene: Scene, position, axes):
    draw_moved_body("RES_T", scene, position, axes)
    data = scene.get_normalized_boundary_normals() * scene.get_resistance_tangential() / 100
    plot_arrows(
        scene.normalized_boundary_nodes + position,
        data,
        axes,
    )


def draw_rectangle(axes, position, scale_x, scale_y):
    axes.add_patch(
        Rectangle(
            (
                position[0],
                position[1],
            ),
            scale_x,
            scale_y,
            fill=None,
            alpha=1.0,
            color="w",
            lw=0.2,
            zorder=3,
        )
    )


def draw_main_displaced(scene: Scene, axes):
    position = np.array([0.0, 0.0])
    draw_displaced(scene, position, "orange", axes)
    for mesh_obstacle in scene.mesh_obstacles:
        draw_displaced(mesh_obstacle, position, "red", axes)

    draw_obstacles(
        scene.linear_obstacle_nodes,
        scene.linear_obstacle_normals,
        position,
        "orange",
        axes,
    )


def draw_base_displaced(scene: Scene, axes):
    position = np.array([0.0, 0.5])  # 1.5
    draw_displaced(scene, position, "purple", axes)


def draw_displaced(scene: Scene, position, color, axes):
    # draw_rectangle(axes, position, scene.mesh_prop.scale_x, scene.mesh_prop.scale_y)
    draw_triplot(scene.moved_nodes + position, scene, f"tab:{color}", axes)
    # draw_data("P", obstacle_forces, scene, [7.5, -1.5], axes)


def draw_nodes(nodes, position, color, axes):
    moved_nodes = nodes + position
    axes.scatter(moved_nodes[:, 0], moved_nodes[:, 1], s=0.1, c=f"tab:{color}")


def draw_forces(scene: Scene, position, axes):
    return draw_data("F", scene.normalized_inner_forces, scene, position, axes)


def draw_input_u(scene: Scene, position, axes):
    return draw_data("U", scene.normalized_displacement_old, scene, position, axes)


def draw_input_v(scene: Scene, position, axes):
    return draw_data("V", scene.rotated_velocity_old, scene, position, axes)


def draw_a(scene, position, axes):
    return draw_data(
        "A",
        scene.exact_acceleration,
        scene,
        position,
        axes,
    )


def draw_data(annotation, data, scene: Scene, position, axes):
    draw_moved_body(annotation, scene, position, axes)
    plot_arrows(scene.normalized_nodes + position, data, axes)


def draw_moved_body(annotation, scene: Scene, position, axes):
    draw_triplot(scene.normalized_nodes + position, scene, "tab:blue", axes)
    add_annotation(annotation, scene, position, axes)


def draw_initial_body(annotation, scene: Scene, position, axes):
    draw_triplot(scene.normalized_initial_nodes + position, scene, "tab:blue", axes)
    add_annotation(annotation, scene, position, axes)


def draw_all_sparse(scene: Scene, position, axes):
    if not hasattr(scene, "all_layers"):
        return
    for i, layer in enumerate(scene.all_layers):
        mesh = layer.mesh
        layer_inner_forces = scene.approximate_boundary_or_all_from_base(
            layer_number=i, base_values=scene.normalized_inner_forces
        )

        triplot(mesh.initial_nodes + position, mesh.elements, color="tab:orange", axes=axes)
        plot_arrows(mesh.initial_nodes + position, layer_inner_forces, axes)
        position[0] += 2.5

        boundary_penetration = scene.get_normalized_boundary_penetration()
        new_boundary_penetration = scene.approximate_boundary_or_all_from_base(
            layer_number=i, base_values=boundary_penetration
        )

        triplot(mesh.initial_nodes + position, mesh.elements, color="tab:blue", axes=axes)
        plot_arrows(mesh.initial_boundary_nodes + position, new_boundary_penetration, axes)
        position[0] += 2.5


def add_annotation(annotation, scene: Scene, position, axes):
    scale = scene.mesh_prop.scale_x
    description_offset = np.array([-0.5, -1.1]) * scale
    axes.annotate(annotation, xy=position + description_offset, color="w", fontsize=5)


def draw_parameters(current_time, scene: Scene, scale, axes):
    x_max = axes.get_xlim()[1]
    y_max = axes.get_ylim()[1]
    args = dict(
        color="w",
        fontsize=5,
    )

    annotation = plotter_common.get_frame_annotation(current_time=current_time, scene=scene)
    axes.text(x_max - 4.0 * scale, y_max - 2.0 * scale, s=annotation, **args)


def draw_triplot(nodes, scene: Scene, color, axes):
    boundary_nodes = nodes[scene.boundary_surfaces]
    axes.add_collection(
        collections.LineCollection(
            boundary_nodes,
            colors=[color for _ in range(boundary_nodes.shape[0])],
            linewidths=0.3,
        )
    )
    triplot(nodes, scene.elements, color, axes)


def triplot(nodes, elements, color, axes):
    axes.triplot(nodes[:, 0], nodes[:, 1], elements, color=color, linewidth=0.1)


def draw_edges_data(position, scene: Scene, axes):
    draw_data_at_edges(scene, scene.edges_data[:, 2:4], position, axes)


def draw_vertices_data(position, scene: Scene, axes):
    draw_data_at_vertices(scene, scene.normalized_displacement_old, position, axes)


def draw_data_at_edges(scene: Scene, features, position, axes):
    draw_triplot(scene.normalized_nodes + position, scene, "tab:orange", axes)

    centers = np.sum(scene.edges_normalized_nodes + position, axis=1) / 2.0
    vertices = scene.edges_normalized_nodes[:, 0] + position
    nodes = (centers + vertices) / 2.0

    for i in range(len(scene.edges_normalized_nodes)):
        feature = np.around(features[i], 2)
        # np.set_printoptions(precision=3)
        axes.text(
            nodes[i, 0] - 0.04,
            nodes[i, 1],
            str(feature),
            # xy=(0.5, 0.5),
            # xycoords="axes fraction",
            fontsize=0.01,
            rotation=30,
            color="w",
        )


def draw_data_at_vertices(scene: Scene, features, position, axes):
    draw_triplot(scene.normalized_nodes + position, scene, "tab:orange", axes)

    nodes = scene.normalized_nodes + position
    for i in range(len(scene.normalized_nodes)):
        feature = np.around(features[i], 2)
        axes.text(
            nodes[i, 0] - 0.04,
            nodes[i, 1],
            str(feature),
            # xy=(0.5, 0.5),
            # xycoords="axes fraction",
            fontsize=0.01,
            rotation=30,
            color="w",
        )


def plot_simple_data(elements, nodes, path):
    fig = get_fig()
    axs = get_axs(fig)
    set_perspective(scale=1, axes=axs)
    triplot(nodes, elements, "tab:orange", axs)
    extension = path.split(".")[-1]
    plotter_common.plt_save(path, extension)
