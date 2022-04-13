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
        config: Config,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            config=config,
            create_in_subprocess=create_in_subprocess,
        )
        self.exact_normalized_a_torch = None  # TODO: clear on change

    def complete_boundary_data_with_zeros_torch(self, data):
        completed_data = torch.zeros((self.nodes_count, data.shape[1]), dtype=data.dtype)
        completed_data[self.boundary_indices] = data
        return completed_data

    @property
    def input_forces_torch(self):
        return thh.to_torch_double(self.input_forces)

    @property
    def normalized_a_correction_torch(self):
        return thh.to_torch_double(self.normalized_a_correction)

    @property
    def const_volume_torch(self):
        return thh.to_torch_double(self.volume)

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
    def normalized_forces_torch(self):
        return thh.to_torch_double(self.normalized_forces)

    @property
    def normalized_displacement_old_torch(self):
        return thh.to_torch_double(self.normalized_displacement_old)

    @property
    def normalized_velocity_old_torch(self):
        return thh.to_torch_double(self.normalized_velocity_old)

    @property
    def input_displacement_old_torch(self):
        return thh.to_torch_double(self.input_displacement_old)

    @property
    def input_velocity_old_torch(self):
        return thh.to_torch_double(self.input_velocity_old)

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
    def normalized_boundary_velocity_old_torch(self):
        return thh.to_torch_double(self.norm_boundary_velocity_old)

    @property
    def normalized_boundary_nodes_torch(self):
        return thh.to_torch_double(self.normalized_boundary_nodes)

    def get_normalized_boundary_normals_torch(self):
        return thh.to_torch_double(self.get_normalized_boundary_normals())

    @property
    def normalized_boundary_obstacle_nodes_torch(self):
        return thh.to_torch_double(self.norm_boundary_obstacle_nodes)

    def get_normalized_boundary_penetration_torch(self):
        return thh.to_torch_double(self.get_normalized_boundary_penetration())

    def get_normalized_boundary_obstacle_normals_torch(self):
        return thh.to_torch_double(self.get_norm_boundary_obstacle_normals())

    def get_normalized_boundary_v_tangential_torch(self):
        return thh.to_torch_double(self.get_normalized_boundary_v_tangential())

    def get_surface_per_boundary_node_torch(self):
        return thh.to_torch_double(self.get_surface_per_boundary_node())

    def get_normalized_E_torch(self):
        return thh.to_torch_double(self.get_normalized_rhs_np(None))
