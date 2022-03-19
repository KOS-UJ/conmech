import torch
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting import SettingRandomized


class SettingTorch(SettingRandomized):
    def __init__(
        self, mesh_data, coefficients, obstacle_coefficients, create_in_subprocess,
    ):
        super().__init__(
            mesh_data, coefficients, obstacle_coefficients, create_in_subprocess,
        )
        self.exact_normalized_a_torch = None  # todo: clear on change

    def complete_boundary_data_with_zeros(self, data):
        completed_data = torch.zeros(
            (self.nodes_count, data.shape[1]), dtype=data.dtype
        )
        completed_data[self.boundary_nodes_indices] = data
        return completed_data

    @property
    def input_u_old_torch(self):
        return thh.to_torch_double(self.input_u_old)

    @property
    def input_v_old_torch(self):
        return thh.to_torch_double(self.input_v_old)

    @property
    def normalized_forces_mean_torch(self):
        return thh.to_torch_double(self.normalized_forces_mean)

    @property
    def predicted_normalized_a_mean_cuda(self):
        return (
            self.normalized_forces_mean_torch.to(thh.device) * self.coefficients.density
        )

    @property
    def input_forces_torch(self):
        return thh.to_torch_double(self.input_forces)

    @property
    def normalized_a_correction_torch(self):
        return thh.to_torch_double(self.normalized_a_correction)

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
    def normalized_boundary_nodes_torch(self):
        return thh.to_torch_double(self.normalized_boundary_nodes)

    @property
    def normalized_boundary_normals_torch(self):
        return thh.to_torch_double(self.normalized_boundary_normals)

    @property
    def normalized_boundary_obstacle_nodes_torch(self):
        return thh.to_torch_double(self.normalized_boundary_obstacle_nodes)

    @property
    def normalized_boundary_obstacle_penetration_vectors_torch(self):
        return thh.to_torch_double(
            self.normalized_boundary_obstacle_penetration_vectors
        )

    @property
    def normalized_boundary_obstacle_normals_torch(self):
        return thh.to_torch_double(self.normalized_boundary_obstacle_normals)

    @property
    def boundary_nodes_volume_torch(self):
        return thh.to_torch_double(self.boundary_nodes_volume)

    @property
    def normalized_E_torch(self):
        return thh.to_torch_double(self.normalized_E)
