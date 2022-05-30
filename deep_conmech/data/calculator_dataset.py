from typing import List

from conmech.scenarios.scenarios import Scenario
from conmech.solvers.calculator import Calculator
from deep_conmech.data.scenario_dataset import ScenariosDataset
from deep_conmech.training_config import TrainingConfig


class CalculatorDataset(ScenariosDataset):
    def __init__(
        self,
        description: str,
        all_scenarios: List[Scenario],
        layers_count: int,
        load_features_to_ram: bool,
        load_targets_to_ram: bool,
        randomize_at_load: bool,
        config: TrainingConfig,
    ):
        super().__init__(
            description=f"{description}_calculator",
            all_scenarios=all_scenarios,
            layers_count=layers_count,
            solve_function=Calculator.solve_all,
            load_features_to_ram=load_features_to_ram,
            load_targets_to_ram=load_targets_to_ram,
            randomize_at_load=randomize_at_load,
            config=config,
        )
