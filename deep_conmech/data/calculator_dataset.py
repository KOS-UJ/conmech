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
        load_data_to_ram: bool,
        with_scenes_file: bool,
        randomize: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
        item_fn=None,
    ):
        super().__init__(
            description=f"{description}_calculator",
            all_scenarios=all_scenarios,
            solve_function=Calculator.solve,  # solve_acceleration_normalized_function,
            load_data_to_ram=load_data_to_ram,
            with_scenes_file=with_scenes_file,
            randomize=randomize,
            config=config,
            rank=rank,
            world_size=world_size,
            item_fn=item_fn,
        )
