import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, collections
from matplotlib.patches import Rectangle
from matplotlib.ticker import LinearLocator

import common.config as config
from simulator.setting.setting_forces import *

background_color = "24292E"  # '1F2428'
plt.rcParams["axes.facecolor"] = background_color
plt.rcParams["figure.facecolor"] = background_color


class Plotter:
    # plt.axes.set_aspect("equal")
    # print(numba.cuda.gpus)

    def draw_animation(self, path, all_images_paths):

        images = []
        for image_path in all_images_paths:
            images.append(imageio.imread(image_path))

        duration = config.PRINT_SKIP
        args = {"duration": duration}
        imageio.mimsave(path, images, **args)

        for image_path in all_images_paths:
            # if(filename.startswith("X") | deleteJpg):
            os.remove(image_path)

    ###########################

    def get_multiple_axs(self, rows, columns):
        plt.figure()
        fig, axs = plt.subplots(rows, columns)
        # fig.patch.set_facecolor('white')
        # fig.patch.set_alpha(0.6)
        return axs

    def get_one_ax(self):
        return plt.axes()

    def draw_setting_ax(self, setting, ax, boundaries, base_setting, time):
        ax.set_aspect("equal", "box")
        scale = setting.scale
        min = np.array([0.0, 0.0])
        max = np.array([scale, scale])

        x_min = -3.0 * 2.0  # min[0] - (boundaries[0] * scale)
        x_max = 13.0 * 2.0  # max[0] + (boundaries[2] * scale)
        y_min = -3.0 * 2.0  # min[1] - (boundaries[3] * scale)
        y_max = 2.0 * 2.0  # max[1] + (boundaries[1] * scale)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        self.draw_main_displaced(setting, ax)
        # self.draw_obstacle_resistance_main(setting, ax)
        if base_setting is not None:
            self.draw_base_displaced(base_setting, scale, ax)
        self.description_offset = np.array([-0.1, -1.1]) * scale

        if config.DRAW_DETAILED:  # detailed:
            self.draw_parameters(time, setting, scale, x_max, y_max, ax)
            # self.draw_angles(setting, ax)
            position = np.array([-1.5, -1.5]) * scale
            if setting.obstacles is not None:
                self.draw_obstacle_resistance_normalized(setting, position, ax)
                position[0] += 2.5 * scale
            # self.draw_boundary_normals(setting, position, ax)
            # position[0] += 2.5 * scale
            self.draw_forces(setting, position, ax)
            position[0] += 2.5 * scale
            # self.draw_u(setting, position, ax)
            # position[0] += 2.0
            self.draw_input_u(setting, position, ax)
            position[0] += 2.5 * scale
            # self.draw_v_n(setting, position, ax)
            # position[0] += 1.5
            # self.draw_v_t(setting, position, ax)
            # position[0] += 1.5
            # self.draw_v(setting, position, ax)
            # position[0] += 2.0
            self.draw_input_v(setting, position, ax)
            position[0] += 2.5 * scale
            self.draw_a(setting, position, ax)

            # self.draw_edges_data(setting, position, ax)
            # self.draw_vertices_data(setting, position, ax)

    def draw_obstacles(self, obstacle_origins, obstacle_normals, position, ax):
        obstacles_tangient = np.hstack(
            (-obstacle_normals[:, 1, None], obstacle_normals[:, 0, None])
        )
        for i in range(len(obstacle_origins)):
            bias = obstacle_origins[i] + position
            ax.arrow(
                *bias,
                *obstacle_normals[i],
                color="tab:gray",
                width=0.00002,
                length_includes_head=True,
                head_width=0.01,
            )
            line = ax.axline(
                bias, obstacles_tangient[i] + bias, color="tab:gray", linewidth=0.4
            )
            # ax.fill_between(bias, obstacles_tangient[i] + bias)

    def draw_main_obstacles(self, setting, ax):
        self.draw_obstacles(
            setting.obstacle_origins, setting.obstacle_normals, [0, 0], ax
        )

    def draw_normalized_obstacles(self, setting, position, ax):
        self.draw_obstacles(
            setting.normalized_obstacle_origins,
            setting.normalized_obstacle_normals,
            position,
            ax,
        )

    def draw_obstacle_resistance_main(self, setting, ax):
        self.draw_arrows(
            setting.boundary_centers, setting.boundary_centers_penetration, ax
        )

    def draw_obstacle_resistance_normalized(self, setting, position, ax):
        self.draw_normalized_obstacles(setting, position, ax)

        self.draw_additional_setting("O", setting, position, ax)
        self.draw_arrows(
            setting.normalized_boundary_centers + position,
            setting.normalized_boundary_centers_penetration,
            ax,
        )

    def draw_boundary_normals(self, setting, position, ax):
        self.draw_additional_setting("N", setting, position, ax)
        self.draw_arrows(
            setting.normalized_boundary_centers + position,
            setting.normalized_boundary_edges_normals,
            ax,
        )

    def draw_rectangle(self, ax, position, scale):
        ax.add_patch(
            Rectangle(
                (position[0], position[1],),
                scale,
                scale,
                fill=None,
                alpha=1.0,
                color="w",
                lw=0.2,
                zorder=3,
            )
        )

    def draw_main_displaced(self, setting, ax):
        position = np.array([0.0, 0.0])
        self.draw_displaced(setting, position, "orange", ax)
        self.draw_points(setting.moved_reference_points, position, "orange", ax)
        if setting.obstacles is not None:
            self.draw_obstacles(
                setting.obstacle_origins, setting.obstacle_normals, position, ax
            )
        """
        self.draw_arrows(
            setting.boundary_centers + position,
            setting.boundary_edges_normals,
            ax,
        )
        self.draw_points(setting.moved_points[setting.boundary_edges_internal_nodes], position, "purple", ax)
        """

    def draw_base_displaced(self, setting, scale, ax):
        position = np.array([0.0, 1.5]) * scale
        self.draw_displaced(setting, position, "purple", ax)
        if setting.obstacles is not None:
            self.draw_obstacles(
                setting.obstacle_origins, setting.obstacle_normals, position, ax
            )

    def draw_displaced(self, setting, position, color, ax):
        self.draw_rectangle(ax, position, setting.scale)
        self.draw_triplot(setting.moved_points + position, setting, f"tab:{color}", ax)
        # self.draw_data("P", obstacle_forces, setting, [7.5, -1.5], ax)

    def draw_points(self, points, position, color, ax):
        moved_points = points + position
        ax.scatter(moved_points[:, 0], moved_points[:, 1], s=0.1, c=f"tab:{color}")

    def draw_v_n(self, setting, position, ax):
        normalized_v_n_old, _ = setting.normalized_v_nt_old
        return self.draw_data("V_N", normalized_v_n_old, setting, position, ax)

    def draw_v_t(self, setting, position, ax):
        _, normalized_v_t_old = setting.normalized_v_nt_old
        return self.draw_data("V_T", normalized_v_t_old, setting, position, ax)

    def draw_forces(self, setting, position, ax):
        return self.draw_data("F", setting.normalized_forces, setting, position, ax)

    def draw_u(self, setting, position, ax):
        return self.draw_data("U", setting.normalized_u_old, setting, position, ax)

    def draw_input_u(self, setting, position, ax):
        return self.draw_data("U_IN", setting.input_u_old, setting, position, ax)

    def draw_v(self, setting, position, ax):
        return self.draw_data("V", setting.normalized_v_old, setting, position, ax)

    def draw_input_v(self, setting, position, ax):
        return self.draw_data("V_IN", setting.input_v_old, setting, position, ax)

    def draw_a(self, setting, position, ax):
        return self.draw_data(
            "A * ts", setting.normalized_a_old * config.TIMESTEP, setting, position, ax
        )

    def draw_data(self, annotation, data, setting, position, ax):
        self.draw_additional_setting(annotation, setting, position, ax)
        self.draw_arrows(setting.normalized_points + position, data, ax)

    def draw_additional_setting(self, annotation, setting, position, ax):
        self.draw_triplot(setting.normalized_points + position, setting, "tab:blue", ax)
        ax.annotate(
            annotation, xy=position + self.description_offset, color="w", fontsize=8
        )

    def draw_edges_data(self, position, setting, ax):
        self.draw_data_edges(setting, setting.edges_data[:, 2:4], position, ax)

    def draw_vertices_data(self, position, setting, ax):
        self.draw_data_vertices(setting, setting.normalized_u_old, position, ax)

    def draw_parameters(self, time, setting, scale, x_max, y_max, ax):
        ax.annotate(
            f"time: {str(round(time, 1))}",
            xy=(x_max - 1.8 * scale, y_max - 0.3 * scale),
            color="w",
            fontsize=5,
        )
        ax.annotate(
            f"nodes: {str(setting.points_number)}",
            xy=(x_max - 1.8 * scale, y_max - 0.6 * scale),
            color="w",
            fontsize=5,
        )

    def draw_angles(self, setting, ax):
        scale = 10.0
        up_vector = setting.up_vector

        start_up = [0.0, 1.0]
        ax.arrow(
            start_up[0],
            start_up[1],
            up_vector[0] / scale,
            up_vector[1] / scale,
            width=0.0002,
            length_includes_head=True,
            head_width=0.01,
        )

        ax.annotate(str(round(setting.angle, 4)), xy=(0.5, 1.5))

    """
    def draw_boundary_arrows(self, setting, normalized_vectors, position, ax):
        scale = 1.0
        points = setting.normalized_boundary_points + position
        scaled_arrows = normalized_vectors * scale

        for i in range(setting.points_number):
            ax.arrow(
                points[i, 0],
                points[i, 1],
                scaled_arrows[i, 0],
                scaled_arrows[i, 1],
                width=0.00005,
                length_includes_head=True,
                head_width=0.02,
                color="w",
                zorder=2,
            )
    """

    def draw_arrows(self, points, normalized_vectors, ax):
        points_number = len(points)
        scale = 1.0  # 3.0
        points = points
        scaled_arrows = normalized_vectors * scale

        max_arrow_count = 64
        arrow_skip = 1
        if points_number > max_arrow_count:
            arrow_skip = int(points_number / max_arrow_count)

        for i in range(points_number):
            if i % arrow_skip == 0:
                ax.arrow(
                    points[i, 0],
                    points[i, 1],
                    scaled_arrows[i, 0],
                    scaled_arrows[i, 1],
                    width=0.00005,
                    length_includes_head=True,
                    head_width=0.02,
                    color="w",
                    zorder=2,
                )

    def draw_triplot(self, points, setting, color, ax):
        boundary_points = points[setting.boundary_edges]
        ax.add_collection(
            collections.LineCollection(
                boundary_points,
                colors=[color for _ in range(boundary_points.shape[0])],
                linewidths=0.3,
            )
        )
        ax.triplot(
            points[:, 0], points[:, 1], setting.cells, color=color, linewidth=0.1
        )

    def plt_save(self, path, extension):
        plt.savefig(
            path,
            transparent=False,
            bbox_inches="tight",
            format=extension,
            pad_inches=0.1,
            dpi=800,  # 1200,
        )
        plt.close()

    ##############

    def draw_data_edges(self, setting, features, position, ax):
        self.draw_triplot(
            setting.normalized_points + position, setting, "tab:orange", ax
        )

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

    def draw_data_vertices(self, setting, features, position, ax):
        self.draw_triplot(
            setting.normalized_points + position, setting, "tab:orange", ax
        )

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

    def draw_colors_triangles(self, mesh, data):
        vertices_number = mesh.cells_points.shape[1]
        centers = np.sum(mesh.cells_points, axis=1) / vertices_number

        colors = np.array([(x - 0.5) ** 2 + (y - 0.5) ** 2 for x, y in centers])
        plt.tripcolor(
            mesh.moved_points[:, 0],
            mesh.moved_points[:, 1],
            mesh.cells,
            facecolors=colors,
            edgecolors="k",
        )
        plt.gca().set_aspect("equal")

        ts = time.time()
        self.plt_save(f"draw_colors_triangles {ts}")


###################


def draw_multiple_mesh_density():
    for id in range(20):
        draw_mesh_density(id)


def draw_mesh_density(id):
    mesh_density = config.MESH_SIZE_PRINT
    corners = config.VAL_PRINT_CORNERS
    # corner_data = np.random.normal(loc=0.0, scale=scale * 0.5, size=4)
    # r = data.interpolate_point(np.array([1,1]), corner_data, corners)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    min = basic_helpers.min(corners)
    max = basic_helpers.max(corners)
    precision = 0.01
    X = np.arange(min[0], max[0], precision)
    Y = np.arange(min[1], max[1], precision)
    X, Y = np.meshgrid(X, Y)
    # points = np.stack((X,Y), axis = -1)

    base_density = basic_helpers.get_base_density(mesh_density, corners)
    corner_data = basic_helpers.mesh_corner_data(base_density)
    Z = basic_helpers.get_adaptive_mesh_density(X, Y, base_density, corner_data)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    max_z = 0.1  # np.max(Z)
    ax.set_zlim(0.0, max_z)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()
    format = "png"
    plt.savefig(
        f"./meshes/mesh_density_{id}.{format}",
        transparent=False,
        bbox_inches="tight",
        format=format,
        pad_inches=0.1,
        dpi=800,  # 1200,
    )
    plt.close()


###################


"""
    def draw_one(self, model, description):
        fig, axs = plt.subplots(1, 1)
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(0.6)
        self.draw_subplot(model, config.DRAW_FORCE_ONE, axs)
        self.plt_save(f"draw_one {description}")

    def draw_nine(self, model, description):
        # dmsh.helpers.show(X, cells, geo)
        # meshio.Mesh(X, {"triangle": cells}).write("circle.vtk")
        # print("Drawing")

        unit = config.DRAW_FORCE_UNIT
        fig, axs = plt.subplots(3, 3)
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(0.6)
        self.draw_subplot(model, [-unit, unit], axs[0, 0])
        self.draw_subplot(model, [0, unit], axs[0, 1])
        self.draw_subplot(model, [unit, unit], axs[0, 2])

        self.draw_subplot(model, [-unit, 0], axs[1, 0])
        self.draw_subplot(model, [0, 0], axs[1, 1])
        self.draw_subplot(model, [unit, 0], axs[1, 2])

        self.draw_subplot(model, [-unit, -unit], axs[2, 0])
        self.draw_subplot(model, [0, -unit], axs[2, 1])
        self.draw_subplot(model, [unit, -unit], axs[2, 2])

        self.plt_save(f"draw_nine {description}")

    def draw_subplot(self, model, force, ax):
        mesh = Mesh(mesh_type=config.MESH_TYPE, mesh_density=config.MESH_SIZE_PRINT)

        self.draw_mesh_ax(self, mesh, ax)

        X = self.get_x_draw_torch(force, mesh)
        u = self.predict(mesh, model, X)
        points = mesh.moved_points + u
        ax.triplot(
            points[:, 0], points[:, 1], mesh.cells, color="tab:orange",
        )

    def get_x_draw_torch(self, force, mesh):
        u_old = np.repeat([[0.0, 0.0]], mesh.points_number, axis=0)
        forces = np.repeat([force], mesh.points_number, axis=0)
        x = np.hstack((forces, u_old, mesh.on_gamma_d))
        x =basic_helpers.to_torch_float(x)
        return x
"""
