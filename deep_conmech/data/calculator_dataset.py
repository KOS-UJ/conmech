import copy
from ctypes import ArgumentError
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
        use_jax: bool,
        all_scenarios: List[Scenario],
        load_data_to_ram: bool,
        with_scenes_file: bool,
        randomize: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
        device_count: int,
        item_fn=None,
    ):
        if config.sc.mode in ["normal", "net", "compare_net"]:
            solve_function = Calculator.solve
        elif config.sc.mode == "skinning":
            solve_function = Calculator.solve_skinning
        else:
            raise ArgumentError()

        super().__init__(
            description=f"{description}_calculator",
            use_jax=use_jax,
            all_scenarios=all_scenarios,
            solve_function=solve_function,
            load_data_to_ram=load_data_to_ram,
            with_scenes_file=with_scenes_file,
            randomize=randomize,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
            item_fn=item_fn,
        )
