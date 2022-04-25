import numpy as np

from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.scene import Scene


class SceneRandomized(Scene):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        config,
        create_in_subprocess: bool,
        with_schur: bool = True,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            normalize_by_rotation=config.normalize_rotate,
            create_in_subprocess=create_in_subprocess,
            with_schur=with_schur,
        )
        self.config = config
        self.velocity_old_randomization = None
        self.displacement_old_randomization = None
        self.randomized_inputs = None

        self.set_randomization(False)
        # printer.print_setting_internal(self, f"output/setting_{helpers.get_timestamp()}.png",
        # None, "png", 0)

    # def remesh(self):
    #    super().remesh()
    #    self.set_randomization(self.randomized_inputs)

    def set_randomization(self, randomized_inputs):
        self.randomized_inputs = randomized_inputs
        if randomized_inputs:
            self.velocity_old_randomization = nph.generate_normal(
                rows=self.nodes_count,
                columns=self.dimension,
                scale=self.config.td.velocity_in_random_factor,
            )
            self.displacement_old_randomization = nph.generate_normal(
                rows=self.nodes_count,
                columns=self.dimension,
                scale=self.config.td.displacement_in_random_factor,
            )
            # Do not randomize boundaries
            self.velocity_old_randomization[self.boundary_indices] = 0
            self.velocity_old_randomization[self.boundary_indices] = 0
        else:
            self.velocity_old_randomization = np.zeros_like(self.initial_nodes)
            self.displacement_old_randomization = np.zeros_like(self.initial_nodes)

    @property
    def normalized_velocity_old_randomization(self):
        return self.normalize_rotate(self.velocity_old_randomization)

    @property
    def normalized_displacement_old_randomization(self):
        return self.normalize_rotate(self.displacement_old_randomization)

    @property
    def randomized_velocity_old(self):
        return self.velocity_old + self.velocity_old_randomization

    @property
    def randomized_displacement_old(self):
        return self.displacement_old + self.displacement_old_randomization

    @property
    def input_velocity_old(self):  # normalized_randomized_velocity_old
        return self.normalized_velocity_old + self.normalized_velocity_old_randomization

    @property
    def input_displacement_old(self):  # normalized_randomized_displacement_old
        return self.normalized_displacement_old + self.normalized_displacement_old_randomization

    @property
    def a_correction(self):
        u_correction = self.config.td.displacement_to_velocity_noise * (
            self.displacement_old_randomization / (self.time_step**2)
        )
        v_correction = (
            (1.0 - self.config.td.displacement_to_velocity_noise)
            * self.velocity_old_randomization
            / self.time_step
        )
        return -1.0 * (u_correction + v_correction)

    @property
    def normalized_a_correction(self):
        return self.normalize_rotate(self.a_correction)

    def make_dirty(self):
        self.velocity_old = self.randomized_velocity_old
        self.displacement_old = self.randomized_displacement_old

        self.velocity_old_randomization = np.zeros_like(self.initial_nodes)
        self.displacement_old_randomization = np.zeros_like(self.initial_nodes)
        self.randomized_inputs = False

    def iterate_self(self, acceleration, temperature=None, randomized_inputs=False):
        _ = temperature
        self.set_randomization(randomized_inputs)
        super().iterate_self(acceleration)
