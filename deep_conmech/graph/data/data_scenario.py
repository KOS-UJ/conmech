from deep_conmech.graph.data.data_base import *
from deep_conmech.common import config, trh
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.graph.setting.setting_input import SettingInput


def get_setting(scenario):
    setting = SettingInput(
        mesh_type=scenario.mesh_type,
        mesh_density_x=scenario.mesh_density,
        mesh_density_y=scenario.mesh_density,
        scale_x=scenario.scale,
        scale_y=scenario.scale,
        is_adaptive=scenario.is_adaptive,
        create_in_subprocess=False,  ###################################
    )
    setting.set_obstacles(scenario.obstacles)
    return setting


def generate_scenario_data_process(dataset, all_scenarios, num_workers, process_id):
    assigned_scenarios = get_assigned_scenarios(all_scenarios, num_workers, process_id)
    data_count = config.VAL_PRINT_EPISODE_STEPS * len(assigned_scenarios)

    start_index = process_id * data_count
    stop_index = start_index + data_count

    indices_to_do = get_indices_to_do(range(start_index, stop_index), dataset.path)
    if not indices_to_do:
        step_tqdm = thh.get_tqdm(
            range(1), desc=f"Process {process_id} - done", position=process_id
        )
        return True

    current_index = start_index
    for scenario in assigned_scenarios:
        scenario_start_index = current_index
        tqdm_description = f"Process {process_id}: Generating {dataset.relative_path} {scenario.id} data"
        step_tqdm = thh.get_tqdm(
            range(1, config.VAL_PRINT_EPISODE_STEPS + 1),
            desc=tqdm_description,
            position=process_id,
        )

        setting = get_setting(scenario)
        for ts in step_tqdm:
            if is_memory_overflow(step_tqdm, tqdm_description):
                step_tqdm.set_description(f"{step_tqdm.desc} - memory overflow")
                return False

            current_time = ts * config.TIMESTEP
            forces = setting.get_forces_by_function(
                scenario.forces_function, current_time
            )
            setting.prepare(forces)

            a, normalized_a = Calculator.solve_all(setting)
            exact_normalized_a_torch = thh.to_torch_double(normalized_a)

            dataset.save(setting, exact_normalized_a_torch, current_index)
            dataset.check_and_print(
                start_index=scenario_start_index,
                current_index=current_index,
                setting=setting,
                step_tqdm=step_tqdm,
                tqdm_description=tqdm_description,
            )

            # setting = setting.get_copy()
            setting.iterate_self(a)
            current_index += 1

    step_tqdm.set_description(f"{step_tqdm.desc} - done")
    return True


class TrainingScenariosDatasetDynamic(BaseDatasetDynamic):
    def __init__(self, base_scenarios, repetitions=2):
        self.all_scenarios = base_scenarios * repetitions
        data_count = config.VAL_PRINT_EPISODE_STEPS * len(self.all_scenarios)
        super().__init__(
            relative_path="training_scenarios",
            data_count=data_count,
            randomize_at_load=True,
        )
        self.repetitions = repetitions
        self.initialize_data()

    def generate_all_data(self):
        # randomly choose scenario parameters
        num_workers = config.GENERATION_WORKERS
        # for process_id in range(num_workers):
        #    generate_scenario_data_process(self, self.all_scenarios, num_workers=num_workers, queue=None, process_id=process_id)

        result = False
        while result is False:
            result = thh.run_processes(
                generate_scenario_data_process, (self, self.all_scenarios), num_workers,
            )
            if result is False:
                print("Restarting data generation")


class ValidationDatasetDynamic(BaseDatasetDynamic):
    def __init__(self, scenario):
        super().__init__(
            relative_path=f"validation/{scenario.id}",
            data_count=config.VAL_PRINT_EPISODE_STEPS,
            randomize_at_load=False,
        )
        self.scenario = scenario
        self.initialize_data()

    def generate_all_data(self):
        result = False
        while result is False:
            result = thh.run_processes(
                generate_scenario_data_process, (self, [self.scenario]), 1,
            )
