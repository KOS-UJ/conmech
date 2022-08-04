import copy
from typing import List

import numpy as np
import torch

from conmech.helpers import nph
from conmech.scenarios.scenarios import Scenario
from conmech.solvers.calculator import Calculator
from deep_conmech.data.scenario_dataset import ScenariosDataset
from deep_conmech.helpers import thh
from deep_conmech.training_config import TrainingConfig


class CalculatorDataset(ScenariosDataset):
    def __init__(
        self,
        description: str,
        all_scenarios: List[Scenario],
        layers_count: int,
        randomize_at_load: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
    ):
        super().__init__(
            description=f"{description}_calculator",
            all_scenarios=all_scenarios,
            layers_count=layers_count,
            solve_function=Calculator.solve_all,  # solve_all_acceleration_normalized_function,  # Calculator.solve_all,
            randomize_at_load=randomize_at_load,
            config=config,
            rank=rank,
            world_size=world_size,
        )

    # def reset(self):
    #     self.all_acceleration = None

    # def update(self, all_acceleration):
    #     self.all_acceleration = all_acceleration

    # def get_state(self, acceleration, prev_scene):
    #     time_step = prev_scene.time_step
    #     velocity = prev_scene.velocity_old + time_step * acceleration
    #     displacement = prev_scene.displacement_old + time_step * velocity
    #     return velocity, displacement

    # def __getitem__(self, index: int):
    #     init_layer_list, init_target_data, init_scene = super().__getitem__(index)
    #     if index == 0 or self.all_acceleration is None:
    #         return init_layer_list, init_target_data

    #     scene = copy.deepcopy(init_scene)

    #     _, _, prev_scene = super().__getitem__(index-1)
    #     velocity, displacement = self.get_state(acceleration=prev_scene.exact_acceleration, prev_scene=prev_scene)
    #     assert np.all(np.equal(scene.velocity_old, velocity))
    #     assert np.all(np.equal(scene.displacement_old, displacement))

    #     acceleration = thh.to_np_double(self.all_acceleration[index-1])
    #     velocity, displacement = self.get_state(acceleration=acceleration, prev_scene=prev_scene)

    #     scene.set_displacement_old(displacement)
    #     scene.set_velocity_old(velocity)
    #     scene.set_acceleration_old(acceleration)

    #     layer_list = [scene.get_features_data()]
    #     return layer_list, init_target_data
