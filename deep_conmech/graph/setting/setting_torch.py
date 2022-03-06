from deep_conmech.simulator.setting.setting_obstacles import *
from deep_conmech.graph.helpers import thh


def L2_torch(argument, C, E):
    first = 0.5 * (C @ argument) - E
    value = first.reshape(-1) @ argument
    return value


class SettingTorch(SettingObstacle):
    def __init__(
        self,
        mesh_type,
        mesh_density_x,
        mesh_density_y,
        scale_x,
        scale_y,
        is_adaptive,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_type,
            mesh_density_x,
            mesh_density_y,
            scale_x,
            scale_y,
            is_adaptive,
            create_in_subprocess,
        )
        self.exact_normalized_a_torch = None  # todo: clear on change

    @property
    def AREA_torch(self):
        return thh.to_torch_double(self.AREA)

    @property
    def B_torch(self):
        return thh.to_torch_double(self.B)

    @property
    def A_plus_B_times_ts_torch(self):
        return thh.to_torch_double(self.A_plus_B_times_ts)

    @property
    def C_torch(self):
        return thh.to_torch_double(self.C)

    @property
    def initial_nodes_torch(self):
        return thh.to_torch_double(self.initial_nodes)

    @property
    def normalized_initial_nodes_torch(self):
        return thh.to_torch_double(self.normalized_initial_nodes)

    @property
    def normalized_points_torch(self):
        return thh.to_torch_double(self.normalized_points)

    @property
    def normalized_forces_torch(self):
        return thh.to_torch_double(self.normalized_forces)

    @property
    def normalized_u_old_torch(self):
        return thh.to_torch_double(self.normalized_u_old)

    @property
    def normalized_v_old_torch(self):
        return thh.to_torch_double(self.normalized_v_old)

    @property
    def input_u_old_torch(self):
        return thh.to_torch_double(self.input_u_old)

    @property
    def input_v_old_torch(self):
        return thh.to_torch_double(self.input_v_old)

    @property
    def boundary_nodes_count_torch(self):
        return thh.to_torch_long(self.boundary_nodes_count)

    @property
    def boundary_faces_count_torch(self):
        return thh.to_torch_long(self.boundary_faces_count)

    @property
    def boundary_faces_torch(self):
        return thh.to_torch_long(self.boundary_faces)

    @property
    def normalized_boundary_v_old_torch(self):
        return thh.to_torch_double(self.normalized_boundary_v_old)

    @property
    def normalized_boundary_points_torch(self):
        return thh.to_torch_double(self.normalized_boundary_points)

    @property
    def normalized_closest_obstacle_normals_torch(self):
        return thh.to_torch_double(self.normalized_closest_obstacle_normals)

    @property
    def normalized_closest_obstacle_origins_torch(self):
        return thh.to_torch_double(self.normalized_closest_obstacle_origins)

    @property
    def angle_torch(self):
        return thh.to_torch_double(self.angle)

    @property
    def normalized_E_torch(self):
        return thh.to_torch_double(self.normalized_E)
