import torch

from conmech.helpers.config import Config
from deep_conmech.graph.scene.scene_randomized import SceneRandomized
from deep_conmech.helpers import thh


class SceneTorch(SceneRandomized):
    def __init__(
        self,
        mesh_prop,
        body_prop,
        obstacle_prop,
        schedule,
        normalize_by_rotation: bool,
        create_in_subprocess: bool,
        with_schur: bool = True,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
            with_schur=with_schur,
        )
        self.exact_normalized_a_torch = None  # TODO: clear on change

    def complete_boundary_data_with_zeros_torch(self, data):
        completed_data = torch.zeros((self.nodes_count, data.shape[1]), dtype=data.dtype)
        completed_data[self.boundary_indices] = data
        return completed_data

    @property
    def normalized_inner_forces_torch(self):
        return thh.to_torch_double(self.normalized_inner_forces)

    def get_normalized_a_correction_torch(self):
        return thh.to_torch_double(self.get_normalized_a_correction())

    @property
    def elasticity_torch(self):
        return thh.to_torch_double(self.elasticity)

    @property
    def viscosity_torch(self):
        return thh.to_torch_double(self.viscosity)

    @property
    def lhs_torch(self):
        return thh.to_torch_double(self.solver_cache.lhs)

    @property
    def initial_nodes_torch(self):
        return thh.to_torch_double(self.initial_nodes)

    @property
    def normalized_initial_nodes_torch(self):
        return thh.to_torch_double(self.normalized_initial_nodes)

    @property
    def normalized_nodes_torch(self):
        return thh.to_torch_double(self.normalized_nodes)

    @property
    def normalized_displacement_torch(self):
        return thh.to_torch_double(self.normalized_displacement)

    @property
    def normalized_velocity_torch(self):
        return thh.to_torch_double(self.normalized_velocity)

    @property
    def input_displacement_torch(self):
        return thh.to_torch_double(self.input_displacement)

    @property
    def input_velocity_torch(self):
        return thh.to_torch_double(self.input_velocity)

    @property
    def boundary_nodes_count_torch(self):
        return thh.to_torch_long(self.boundary_nodes_count)

    @property
    def boundary_surfaces_count_torch(self):
        return thh.to_torch_long(self.boundary_surfaces_count)

    @property
    def boundary_surfaces_torch(self):
        return thh.to_torch_long(self.boundary_surfaces)

    @property
    def norm_boundary_velocity_torch(self):
        return thh.to_torch_double(self.norm_boundary_velocity)

    @property
    def norm_boundary_nodes_torch(self):
        return thh.to_torch_double(self.normalized_boundary_nodes)

    def get_normalized_boundary_normals_torch(self):
        return thh.to_torch_double(self.get_normalized_boundary_normals())

    @property
    def norm_boundary_obstacle_nodes_torch(self):
        return thh.to_torch_double(self.norm_boundary_obstacle_nodes)

    def get_normalized_boundary_penetration_torch(self):
        return thh.to_torch_double(self.get_normalized_boundary_penetration())

    def get_normalized_boundary_obstacle_normals_torch(self):
        return thh.to_torch_double(self.get_norm_boundary_obstacle_normals())

    def get_normalized_boundary_v_tangential_torch(self):
        return thh.to_torch_double(self.get_normalized_boundary_v_tangential())

    def get_surface_per_boundary_node_torch(self):
        return thh.to_torch_double(self.get_surface_per_boundary_node())

    def get_normalized_rhs_torch(self):
        return thh.to_torch_double(self.get_normalized_rhs_np())
