from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from conmech.helpers.config import Config
from conmech.mesh.mesh import Mesh
from conmech.plotting import plotter_common
from conmech.plotting.plotter_common import PlotAnimationConfig, make_animation
from conmech.scene.scene import Scene
from conmech.scene.scene_temperature import SceneTemperature


def get_fig():
    return plt.figure(figsize=(3, 2))


def get_one_ax(fig, rect, angle, distance):
    # axes = fig.add_subplot(1, 1, 1, projection="3d", facecolor="none")
    axes = fig.add_axes(rect, projection="3d", facecolor="none")
    axes.set_proj_type("ortho")
    axes.view_init(elev=angle[0], azim=angle[1])  # , vertical_axis='y')
    axes.dist = distance

    axes.grid(False)
    axes.w_xaxis.pane.fill = False
    axes.w_yaxis.pane.fill = False
    axes.w_zaxis.pane.fill = False

    aspect = (12, 4, 4)
    axes.set_box_aspect(aspect)

    axes.set_xlim(-1, aspect[0] - 1)
    axes.set_ylim(-1, aspect[1] - 1)
    axes.set_zlim(-1, aspect[2] - 1)

    # axes.set_xlabel("x", labelpad=0.05, color="w")
    # axes.set_ylabel("y", labelpad=0.05, color="w")
    # axes.set_zlabel("z", labelpad=0.05, color="w")

    ticks = []  # np.arange(0, 2, 1)
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)
    axes.set_zticks(ticks)

    # axes.tick_params(axis="x", colors="w")
    # axes.tick_params(axis="y", colors="w")
    # axes.tick_params(axis="z", colors="w")

    axes.w_xaxis.line.set_color("w")
    axes.w_yaxis.line.set_color("w")
    axes.w_zaxis.line.set_color("w")

    return axes


def get_axs(fig):
    angles = np.array([[[30, -60], [0, -90]], [[0, 0], [90, 0]]])
    distances = np.array([[9, 10], [10, 10]])

    ax0 = get_one_ax(fig, [0.0, 0.0, 0.7, 0.7], angles[0, 0], distances[0, 0])
    ax1 = get_one_ax(fig, [0.2, 0.5, 0.4, 0.4], angles[0, 1], distances[0, 0])
    ax2 = get_one_ax(fig, [0.6, 0.5, 0.4, 0.4], angles[1, 0], distances[0, 1])
    ax3 = get_one_ax(fig, [0.6, 0.2, 0.4, 0.4], angles[1, 1], distances[1, 1])

    return [ax0, ax1, ax2, ax3]


def plot_frame(
    fig,
    axs,
    scene: Scene,
    current_time,
    base_scene: Optional[Scene] = None,
    t_scale: Optional[List] = None,
):
    _ = base_scene
    for axes in axs:
        plot_subframe(
            axes=axes,
            scene=scene,
            normalized_data={
                "F": scene.normalized_inner_forces,
                "U": scene.normalized_displacement_old,
                "V": scene.normalized_velocity_old,
                "A": scene.normalized_a_old,
            },
            t_scale=t_scale,
        )
    draw_parameters(axes=axs[0], scene=scene, current_time=current_time)

    if isinstance(scene, SceneTemperature):
        cbar_settings = plotter_common.get_t_data(t_scale=t_scale)
        plotter_common.plot_colorbar(fig, axs=axs, cbar_settings=cbar_settings)


def draw_parameters(axes, scene: Scene, current_time):
    annotation = plotter_common.get_frame_annotation(current_time=current_time, scene=scene)
    x_max = axes.get_xlim()[1]
    y_max = axes.get_ylim()[1]
    z_max = axes.get_zlim()[1]

    args = dict(color="w", fontsize=4)
    axes.text(x_max - 2.0, y_max - 0.5, z_max + 1.0, s=annotation, **args)  # zdir=None,


def plot_arrows(starts, vectors, axes):
    prepared_starts, prepared_vectors = plotter_common.prepare_for_arrows(starts, vectors)
    axes.quiver(*prepared_starts, *prepared_vectors, arrow_length_ratio=0.1, color="w", lw=0.1)


def draw_base_arrows(axes, base):
    z = np.array([0, 2.0, 2.0])
    axes.quiver(*z, *(base[0]), arrow_length_ratio=0.1, color="r")
    axes.quiver(*z, *(base[1]), arrow_length_ratio=0.1, color="y")
    axes.quiver(*z, *(base[2]), arrow_length_ratio=0.1, color="g")


def plot_subframe(axes, scene: Scene, normalized_data: dict, t_scale):
    draw_base_arrows(axes, scene.moved_base)

    if isinstance(scene, SceneTemperature):
        cbar_settings = plotter_common.get_t_data(t_scale)
        plot_main_temperature(
            axes,
            nodes=scene.moved_nodes,
            scene=scene,
            cbar_settings=cbar_settings,
        )
    else:
        plot_mesh(nodes=scene.moved_nodes, mesh=scene, color="tab:orange", axes=axes)
    plot_obstacles(axes, scene, "tab:orange")

    shift = np.array([0, 2.0, 0])
    for key, data in normalized_data.items():
        shifted_normalized_nodes = scene.normalized_nodes + shift
        args = dict(color="w", fontsize=4)
        axes.text(*(shift - 1), s=key, **args)  # zdir=None,
        plot_mesh(nodes=shifted_normalized_nodes, mesh=scene, color="tab:blue", axes=axes)
        plot_arrows(starts=shifted_normalized_nodes, vectors=data, axes=axes)
        shift += np.array([2.5, 0, 0])

    return
    shift = np.array([0, 2.0, 1.5])
    if hasattr(scene, "all_layers"):
        for _, layer in enumerate(scene.all_layers):
            mesh = layer.mesh
            shifted_normalized_nodes = mesh.initial_nodes + shift
            # layer_inner_forces = scene.approximate_boundary_or_all_from_base(
            #     layer_number=i, base_values=scene.normalized_inner_forces
            # )
            # plot_arrows(starts=shifted_normalized_nodes, vectors=layer_inner_forces, axes=axes)
            plot_mesh(nodes=shifted_normalized_nodes, mesh=mesh, color="tab:blue", axes=axes)
            shift += np.array([2.5, 0, 0])

    if isinstance(scene, SceneTemperature):
        plot_temperature(
            axes=axes,
            nodes=shifted_normalized_nodes,
            scene=scene,
            cbar_settings=cbar_settings,
        )


def plot_temperature(axes, nodes, scene: Scene, cbar_settings: plotter_common.ColorbarSettings):
    axes.scatter(
        *(nodes.T),
        c=scene.t_old,
        vmin=cbar_settings.vmin,
        vmax=cbar_settings.vmax,
        cmap=cbar_settings.cmap,
        s=1,
        marker=".",
        linewidths=0.1,
    )


def plot_main_temperature(axes, nodes, scene: Scene, cbar_settings):
    boundary_surfaces_nodes = nodes[scene.boundary_surfaces]
    nodes_temperature = scene.t_old[scene.boundary_surfaces]
    faces_temperature = np.mean(nodes_temperature, axis=1)

    facecolors = cbar_settings.mappable.to_rgba(faces_temperature)
    axes.add_collection3d(
        Poly3DCollection(
            boundary_surfaces_nodes,
            linewidths=0.1,
            facecolors=facecolors,
            alpha=0.2,
        )
    )


def plot_mesh(nodes, mesh: Mesh, color, axes):
    boundary_surfaces_nodes = nodes[mesh.boundary_surfaces]
    axes.add_collection3d(
        Poly3DCollection(
            boundary_surfaces_nodes,
            edgecolors=color,
            linewidths=0.1,
            facecolors=color,
            alpha=0.2,
        )
    )


def plot_obstacles(axes, scene: Scene, color):
    if scene.has_no_obstacles:
        return
    alpha = 0.3
    node = scene.linear_obstacle_nodes[0]
    normal = scene.get_obstacle_normals()[0]

    # a plane is a*x+b*y+c*z+d=0
    # z = (-d-axes-by) / c
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -node.dot(normal)

    x_rng = np.arange(-1.2, 11.2, 0.2)
    y_rng = np.arange(-1.2, 3.2, 0.2)

    X, Y = np.meshgrid(x_rng, y_rng)
    Z = (-normal[0] * X - normal[1] * Y - d) / normal[2]
    mask = (Z > -1.2) * (Z < 3.2)

    axes.plot_surface(X * mask, Y * mask, Z * mask, color=color, alpha=alpha)
    # axes.plot_surface(X[:,col], Y[:,col], Z[:,col], color=color, alpha=alpha)

    axes.quiver(*node, *normal, color=color, alpha=alpha)


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
