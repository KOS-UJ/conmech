import numpy as np

from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from deep_conmech.simulator.setting.setting_obstacles import SettingObstacles
from deep_conmech.training_config import TrainingConfig


class SettingRandomized(SettingObstacles):
    def __init__(
        self,
        mesh_data: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        config: TrainingConfig,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            normalize_by_rotation=config.normalize_rotate,
            create_in_subprocess=create_in_subprocess,
        )
        self.config = config
        self.set_randomization(False)
        # printer.print_setting_internal(self, f"output/setting_{helpers.get_timestamp()}.png", None, "png", 0)

    def remesh(self):
        super().remesh()
        self.set_randomization(self.randomized_inputs)

    def set_randomization(self, randomized_inputs):
        self.randomized_inputs = randomized_inputs
        if randomized_inputs:
            self.v_old_randomization = nph.get_random_normal(
                self.dimension, self.nodes_count, self.config.td.V_IN_RANDOM_FACTOR
            )
            self.u_old_randomization = nph.get_random_normal(
                self.dimension, self.nodes_count, self.config.td.U_IN_RANDOM_FACTOR
            )
            # Do not randomize boundaries
            self.v_old_randomization[self.boundary_indices] = 0
            self.v_old_randomization[self.boundary_indices] = 0
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
    def input_forces(self):
        return self.normalized_forces  # - self.normalized_forces_mean

    @property
    def a_correction(self):
        u_correction = self.config.td.U_NOISE_GAMMA * (
            self.u_old_randomization / (self.time_step**2)
        )
        v_correction = (
            (1.0 - self.config.td.U_NOISE_GAMMA) * self.v_old_randomization / self.time_step
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

    def iterate_self(self, acceleration, randomized_inputs=False):
        self.set_randomization(randomized_inputs)
        super().iterate_self(acceleration)
