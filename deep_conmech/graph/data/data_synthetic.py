import deep_conmech.graph.data.interpolation_helpers as interpolation_helpers
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.schedule import Schedule
from deep_conmech import scenarios
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.solver import Solver


def create_forces(config, setting):
    if interpolation_helpers.decide(config.DATA_ZERO_FORCES):
        forces = np.zeros([setting.nodes_count, setting.dimension])
    else:
        forces = interpolation_helpers.interpolate_four(
            setting.nodes_count,
            setting.initial_nodes,
            config.FORCES_RANDOM_SCALE,
            setting.mesh_data.scale_x,
            setting.mesh_data.scale_y,
        )
    return forces


def create_u_old(config, setting):
    u_old = interpolation_helpers.interpolate_four(
        setting.nodes_count,
        setting.initial_nodes,
        config.U_RANDOM_SCALE,
        setting.mesh_data.scale_x,
        setting.mesh_data.scale_y,
    )
    return u_old


def create_v_old(config, setting):
    if interpolation_helpers.decide(config.DATA_ROTATE_VELOCITY):
        v_old = interpolation_helpers.interpolate_rotate(
            setting.nodes_count,
            setting.initial_nodes,
            config.V_RANDOM_SCALE,
            setting.mesh_data.scale_x,
            setting.mesh_data.scale_y,
        )
    else:
        v_old = interpolation_helpers.interpolate_four(
            setting.nodes_count,
            setting.initial_nodes,
            config.V_RANDOM_SCALE,
            setting.mesh_data.scale_x,
            setting.mesh_data.scale_y,
        )
    return v_old


def create_obstacles(config, setting):
    obstacle_normals_unnormaized = nph.get_random_normal_circle_numba(
        setting.dimension, 1, config.OBSTACLE_ORIGIN_SCALE
    )
    obstacle_origins = -obstacle_normals_unnormaized + setting.mean_moved_nodes
    return np.stack((obstacle_normals_unnormaized, obstacle_origins))


def create_mesh_type():
    return interpolation_helpers.choose(
        ["pygmsh_rectangle", "pygmsh_circle", "pygmsh_spline", "pygmsh_polygon"]
    )


def create_obstacles(config, setting):
    obstacle_normals_unnormaized = nph.get_random_normal_circle_numba(
        setting.dimension, 1, config.OBSTACLE_ORIGIN_SCALE
    )
    obstacle_origins = -obstacle_normals_unnormaized + setting.mean_moved_nodes
    return np.stack((obstacle_normals_unnormaized, obstacle_origins))


def get_base_setting(config, mesh_type):
    return SettingInput(
        mesh_data=MeshData(
            mesh_type=mesh_type,
            mesh_density=[config.MESH_DENSITY],
            scale=[config.TRAIN_SCALE],
            is_adaptive=config.ADAPTIVE_TRAINING_MESH,
        ),
        body_prop=scenarios.default_body_prop,
        obstacle_prop=scenarios.default_obstacle_prop,
        schedule=Schedule(final_time=config.FINAL_TIME),
        config=config,
        create_in_subprocess=False,
    )


class TrainingSyntheticDatasetDynamic(BaseDatasetDynamic):
    def __init__(self, config, dimension):
        num_workers = config.GENERATION_WORKERS
        data_count = config.SYNTHETIC_SOLVERS_COUNT

        if data_count % num_workers != 0:
            raise Exception("Cannot divide data generation work")
        self.data_part_count = int(data_count / num_workers)

        super().__init__(
            dimension=dimension,
            relative_path="training_synthetic",
            data_count=data_count,
            randomize_at_load=True,
            num_workers=num_workers,
            config=config
        )
        self.initialize_data()

    def generate_setting(self, index):
        mesh_type = create_mesh_type()
        setting = get_base_setting(self.config, mesh_type)
        # setting.set_randomization(True)

        obstacles_unnormaized = create_obstacles(setting)
        forces = create_forces(self.config, setting)
        u_old = create_u_old(self.config, setting)
        v_old = create_v_old(self.config, setting)

        setting.set_obstacles(obstacles_unnormaized)
        setting.set_u_old(u_old)
        setting.set_v_old(v_old)
        setting.prepare(forces)

        add_label = False
        if add_label:
            normalized_a = Solver.solve_normalized(setting)
            exact_normalized_a_torch = thh.to_torch_double(normalized_a)
        else:
            exact_normalized_a_torch = None

        # data = setting.get_data(index, exact_normalized_a_torch)
        return setting, exact_normalized_a_torch  # data, setting

    def generate_data_process(self, num_workers, process_id):
        assigned_data_range = get_process_data_range(process_id, self.data_part_count)

        indices_to_do = get_and_check_indices_to_do(
            assigned_data_range, self.path, process_id
        )
        if not indices_to_do:
            return True

        tqdm_description = (
            f"Process {process_id} - generating {self.relative_path} data"
        )
        step_tqdm = cmh.get_tqdm(
            indices_to_do, desc=tqdm_description, config=self.config, position=process_id,
        )
        for index in step_tqdm:
            if is_memory_overflow(step_tqdm, tqdm_description):
                return False

            setting, exact_normalized_a_torch = self.generate_setting(index)
            self.save(setting, exact_normalized_a_torch, index)
            self.check_and_print(
                len(indices_to_do), index, setting, step_tqdm, tqdm_description
            )

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True
