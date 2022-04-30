import numpy as np

from conmech.helpers import nph
from conmech.scene.scene import Scene
from deep_conmech.training_config import TrainingConfig


class SceneRandomized:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.velocity_in_random_factor = 0
        self.displacement_in_random_factor = 0
        self.displacement_to_velocity_noise = 0
        self.velocity_randomization = np.zeros_like(self.initial_nodes)
        self.displacement_randomization = np.zeros_like(self.initial_nodes)
        # printer.print_setting_internal(self, f"output/setting_{helpers.get_timestamp()}.png",
        # None, "png", 0)

    def __getattr__(self, name):
        return getattr(self.scene, name)

    # def remesh(self):
    #    super().remesh()
    #    self.set_randomization(self.randomized_inputs)
    @property
    def randomized_inputs(self):
        return self.displacement_in_random_factor == 0 and self.displacement_to_velocity_noise == 0

    def unset_randomization(self):
        self.velocity_in_random_factor = 0
        self.displacement_in_random_factor = 0
        self.displacement_to_velocity_noise = 0
        self.regenerate_randomization()

    def set_randomization(self, config: TrainingConfig):
        self.velocity_in_random_factor = config.td.velocity_in_random_factor
        self.displacement_in_random_factor = config.td.displacement_in_random_factor
        self.displacement_to_velocity_noise = config.td.displacement_to_velocity_noise
        self.regenerate_randomization()

    def regenerate_randomization(self):
        self.velocity_randomization = nph.generate_normal(
            rows=self.nodes_count,
            columns=self.dimension,
            scale=self.velocity_in_random_factor,
        )
        self.displacement_randomization = nph.generate_normal(
            rows=self.nodes_count,
            columns=self.dimension,
            scale=self.displacement_in_random_factor,
        )
        # Do not randomize boundaries
        self.displacement_randomization[self.boundary_indices] = 0.0
        self.velocity_randomization[self.boundary_indices] = 0.0

    @property
    def randomized_velocity(self):
        return self.velocity + self.velocity_randomization

    @property
    def randomized_displacement(self):
        return self.displacement + self.displacement_randomization

    @property
    def norm_rand_velocity(self):
        return self.normalize_shift_and_rotate(self.velocity + self.velocity_randomization)

    @property
    def norm_rand_displacement(self):
        return self.normalize_shift_and_rotate(self.displacement + self.displacement_randomization)

    @property
    def input_velocity(self):
        return self.norm_rand_velocity

    @property
    def input_displacement(self):
        return self.norm_rand_displacement

    def get_a_correction(self):
        u_correction = self.displacement_to_velocity_noise * (
            self.displacement_randomization / (self.time_step**2)
        )
        v_correction = (
            (1.0 - self.displacement_to_velocity_noise)
            * self.velocity_randomization
            / self.time_step
        )
        return -1.0 * (u_correction + v_correction)

    def get_normalized_a_correction(self):
        return self.normalize_rotate(self.get_a_correction())

    def make_dirty(self):
        self.velocity = self.randomized_velocity
        self.displacement = self.randomized_displacement

        self.unset_randomization()

    def iterate_self(self, acceleration, temperature=None):
        _ = temperature
        if self.randomized_inputs:
            self.regenerate_randomization()
        self.scene.iterate_self(acceleration)
