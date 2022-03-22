import copy
from conmech.dataclass.body_properties import BodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.obstacle_properties import ObstacleProperties
from conmech.dataclass.time_data import TimeData

import deep_conmech.simulator.mesh.remesher as remesher
from conmech.helpers import nph
from deep_conmech.common import config
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_obstacles import SettingObstacles
from deep_conmech.scenarios import Scenario


class SettingRandomized(SettingObstacles):
    def __init__(
        self,
        mesh_data: MeshData,
        body_prop: BodyProperties,
        obstacle_prop: ObstacleProperties,
        time_data: TimeData,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            time_data=time_data,
            create_in_subprocess=create_in_subprocess,
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
                self.dim, self.nodes_count, config.V_IN_RANDOM_FACTOR
            )
            self.u_old_randomization = nph.get_random_normal(
                self.dim, self.nodes_count, config.U_IN_RANDOM_FACTOR
            )
            # Do not randomize boundaries
            self.v_old_randomization[self.boundary_nodes_indices] = 0
            self.v_old_randomization[self.boundary_nodes_indices] = 0
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
    def normalized_forces_mean(self):
        return np.mean(self.normalized_forces, axis=0)

    @property
    def input_forces(self):
        return self.normalized_forces  # - self.normalized_forces_mean

    @property
    def a_correction(self):
        u_correction = config.U_NOISE_GAMMA * (
            self.u_old_randomization / (self.time_step ** 2)
        )
        v_correction = (
            (1.0 - config.U_NOISE_GAMMA) * self.v_old_randomization / self.time_step
        )
        return -1.0 * (u_correction + v_correction)

    @property
    def normalized_a_correction(self):
        return self.normalize_rotate(self.a_correction)

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
        v = self.v_old + self.time_step * a
        u = self.u_old + self.time_step * v

        self.set_randomization(randomized_inputs)
        self.set_u_old(u)
        self.set_v_old(v)
        self.set_a_old(a)

        self.clear()
        return self

    def remesh_self(self):
        old_initial_nodes = self.initial_nodes.copy()
        old_elements = self.elements.copy()
        u_old = self.u_old.copy()
        v_old = self.v_old.copy()
        a_old = self.a_old.copy()

        self.remesh()

        u = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_nodes, u_old, old_elements
        )
        v = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_nodes, v_old, old_elements
        )
        a = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_nodes, a_old, old_elements
        )

        self.set_u_old(u)
        self.set_v_old(v)
        self.set_a_old(a)

    @staticmethod
    def get_setting(
        scenario: Scenario, randomize: bool = False, create_in_subprocess: bool = False
    ):
        setting = SettingRandomized(
            mesh_data=scenario.mesh_data,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            time_data=scenario.time_data,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_randomization(randomize)
        setting.set_obstacles(scenario.obstacles)
        return setting
