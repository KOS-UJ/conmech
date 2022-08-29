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
        load_data_to_ram: bool,
        randomize: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
        item_fn=None,
    ):
        super().__init__(
            description=f"{description}_calculator",
            all_scenarios=all_scenarios,
            layers_count=layers_count,
            solve_function=Calculator.solve_all,  # solve_all_acceleration_normalized_function,  # Calculator.solve_all,
            load_data_to_ram=load_data_to_ram,
            randomize=randomize,
            config=config,
            rank=rank,
            world_size=world_size,
            item_fn=item_fn,
        )
