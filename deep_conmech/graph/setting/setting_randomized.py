import copy

import deep_conmech.simulator.mesh.remesher as remesher
from deep_conmech.common import *
from deep_conmech.graph.setting.setting_torch import *
from deep_conmech.simulator.setting.setting_forces import *
from conmech.helpers import nph


# MIN AT
# a = a_cleaned - ((v_old - randomized_v_old) / config.TIMESTEP
def L2_normalized_correction_cuda(
    cleaned_normalized_a_cuda, C_cuda, normalized_E_cuda, normalized_a_correction_cuda
):
    normalized_a_cuda = cleaned_normalized_a_cuda - normalized_a_correction_cuda
    return L2_normalized_cuda(normalized_a_cuda, C_cuda, normalized_E_cuda)


def L2_normalized_cuda(normalized_a_cuda, C_cuda, normalized_E_cuda):
    normalized_a_vector_cuda = nph.stack_column(normalized_a_cuda)
    value = L2_torch(normalized_a_vector_cuda.double(), C_cuda, normalized_E_cuda,)
    return value


class SettingRandomized(SettingTorch):
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
        self.set_randomization(False)
        # printer.print_setting_internal(self, f"output/setting_{helpers.get_timestamp()}.png", None, "png", 0)

    def remesh(self):
        super().remesh()
        self.set_randomization(self.randomized_inputs)

    def set_randomization(self, randomized_inputs):
        self.randomized_inputs = randomized_inputs
        if randomized_inputs:
            self.v_old_randomization = nph.get_random_normal(
                self.nodes_count, config.V_IN_RANDOM_FACTOR
            )
            self.u_old_randomization = nph.get_random_normal(
                self.nodes_count, config.U_IN_RANDOM_FACTOR
            )
        else:
            self.v_old_randomization = np.zeros_like(self.initial_nodes)
            self.u_old_randomization = np.zeros_like(self.initial_nodes)

    @property
    def normalized_v_old_randomization(self):
        return self.normalize_rotate(self.v_old_randomization)

    @property
    def normalized_u_old_randomization(self):
        return self.normalize_rotate(self.u_old_randomization)

    @property
    def randomized_v_old(self):
        return self.v_old + self.v_old_randomization

    @property
    def randomized_u_old(self):
        return self.u_old + self.u_old_randomization

    @property
    def input_v_old(self):  # normalized_randomized_v_old
        return self.normalized_v_old + self.normalized_v_old_randomization

    @property
    def input_u_old(self):  # normalized_randomized_u_old
        return self.normalized_u_old + self.normalized_u_old_randomization

    @property
    def input_u_old_torch(self):
        return thh.to_torch_double(self.input_u_old)

    @property
    def input_v_old_torch(self):
        return thh.to_torch_double(self.input_v_old)

    @property
    def normalized_forces_mean(self):
        return np.mean(self.normalized_forces, axis=0)

    @property
    def normalized_forces_mean_torch(self):
        return thh.to_torch_double(self.normalized_forces_mean)

    @property
    def predicted_normalized_a_mean_cuda(self):
        return self.normalized_forces_mean_torch.to(thh.device) * config.DENS

    @property
    def input_forces(self):
        return self.normalized_forces  # - self.normalized_forces_mean

    @property
    def input_forces_torch(self):
        return thh.to_torch_double(self.input_forces)

    @property
    def a_correction(self):
        u_correction = config.U_NOISE_GAMMA * (
            self.u_old_randomization / (config.TIMESTEP * config.TIMESTEP)
        )
        v_correction = (
            (1.0 - config.U_NOISE_GAMMA) * self.v_old_randomization / config.TIMESTEP
        )
        return -1.0 * (u_correction + v_correction)

    @property
    def normalized_a_correction(self):
        return self.normalize_rotate(self.a_correction)

    @property
    def normalized_a_correction_torch(self):
        return thh.to_torch_double(self.normalized_a_correction)

    def make_dirty(self):
        self.v_old = self.randomized_v_old
        self.u_old = self.randomized_u_old

        self.v_old_randomization = np.zeros_like(self.initial_nodes)
        self.u_old_randomization = np.zeros_like(self.initial_nodes)
        self.randomized_inputs = False

    def get_copy(self):
        setting = copy.deepcopy(self)
        return setting

    def iterate_self(self, a, randomized_inputs=False):
        v = self.v_old + config.TIMESTEP * a
        u = self.u_old + config.TIMESTEP * v

        self.set_randomization(randomized_inputs)
        self.set_u_old(u)
        self.set_v_old(v)
        self.set_a_old(a)

        self.clear()
        return self

    def remesh_self(self):
        old_initial_points = self.initial_nodes.copy()
        old_cells = self.cells.copy()
        u_old = self.u_old.copy()
        v_old = self.v_old.copy()
        a_old = self.a_old.copy()

        self.remesh()

        u = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_points, u_old, old_cells
        )
        v = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_points, v_old, old_cells
        )
        a = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_points, a_old, old_cells
        )

        self.set_u_old(u)
        self.set_v_old(v)
        self.set_a_old(a)
