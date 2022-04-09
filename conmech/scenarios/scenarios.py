from typing import Callable, Optional, Union

import numpy as np
from conmech.helpers import cmh
from conmech.helpers.config import Config
from conmech.properties.body_properties import (
    DynamicBodyProperties,
    DynamicTemperatureBodyProperties,
)
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import (
    ObstacleProperties,
    TemperatureObstacleProperties,
)
from conmech.properties.schedule import Schedule
from conmech.solvers.calculator import Calculator
from deep_conmech.simulator.setting.setting_obstacles import SettingObstacles
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature
from deep_conmech.training_config import TrainingData


class Scenario:
    def __init__(
        self,
        name: str,
        mesh_data: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        forces_function: Union[Callable[..., np.ndarray], np.ndarray],
        obstacles: Optional[np.ndarray],
    ):
        self.name = name
        self.mesh_data = mesh_data
        self.body_prop = body_prop
        self.obstacle_prop = obstacle_prop
        self.schedule = schedule
        self.obstacles = None if obstacles is None else obstacles * mesh_data.scale_x
        self.forces_function = forces_function

    @staticmethod
    def get_by_function(function, setting, current_time):
        if isinstance(function, np.ndarray):
            return np.tile(function, (setting.nodes_count, 1))
        return np.array(
            [
                function(*nodes_pairs, setting.mesh_data, current_time)
                for nodes_pairs in zip(setting.initial_nodes, setting.moved_nodes)
            ]
        )

    def get_forces_by_function(self, setting, current_time):
        return Scenario.get_by_function(self.forces_function, setting, current_time)

    def get_tqdm(self, desc: str, config: Config):
        return cmh.get_tqdm(
            iterable=range(self.schedule.episode_steps),
            config=config,
            desc=f"{desc} {self.name}",
        )

    @staticmethod
    def get_solve_function():
        return Calculator.solve

    def get_setting(
        self,
        normalize_by_rotation=True,
        randomize=False,
        create_in_subprocess: bool = False,
    ) -> SettingObstacles:
        _ = randomize
        setting = SettingObstacles(
            mesh_data=self.mesh_data,
            body_prop=self.body_prop,
            obstacle_prop=self.obstacle_prop,
            schedule=self.schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        setting.normalize_and_set_obstacles(self.obstacles)
        return setting

    @property
    def dimension(self):
        return self.mesh_data.dimension

    @property
    def time_step(self):
        return self.schedule.time_step

    @property
    def final_time(self):
        return self.schedule.final_time


class TemperatureScenario(Scenario):
    def __init__(
        self,
        name: str,
        mesh_data: MeshProperties,
        body_prop: DynamicTemperatureBodyProperties,
        obstacle_prop: TemperatureObstacleProperties,
        schedule: Schedule,
        forces_function: Union[Callable, np.ndarray],
        obstacles: Optional[np.ndarray],
        heat_function: Union[Callable, np.ndarray],
    ):
        super().__init__(
            name=name,
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            forces_function=forces_function,
            obstacles=obstacles,
        )
        self.heat_function = heat_function

    def get_heat_by_function(self, setting, current_time):
        return Scenario.get_by_function(self.heat_function, setting, current_time)

    @staticmethod
    def get_solve_function():
        return Calculator.solve_with_temperature

    def get_setting(
        self,
        normalize_by_rotation=True,
        randomize=False,
        create_in_subprocess: bool = False,
    ) -> SettingTemperature:
        _ = randomize
        setting = SettingTemperature(
            mesh_data=self.mesh_data,
            body_prop=self.body_prop,
            obstacle_prop=self.obstacle_prop,
            schedule=self.schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        setting.normalize_and_set_obstacles(self.obstacles)
        return setting


default_schedule = Schedule(time_step=0.01, final_time=4.0)

default_body_prop = DynamicBodyProperties(
    mu=4.0, lambda_=4.0, theta=4.0, zeta=4.0, mass_density=1.0
)
# body_prop = DynamicBodyProperties(mu=0.01, lambda_=0.01, theta=0.01, zeta=0.01, mass_density=0.01)

default_thermal_expansion_coefficients = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
default_thermal_conductivity_coefficients = np.array(
    [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
)
default_temp_body_prop = DynamicTemperatureBodyProperties(
    mass_density=1.0,
    mu=4.0,
    lambda_=4.0,
    theta=4.0,
    zeta=4.0,
    thermal_expansion=default_thermal_expansion_coefficients,
    thermal_conductivity=default_thermal_conductivity_coefficients,
)


def get_temp_body_prop(thermal_expansion_coeff, thermal_conductivity_coeff):
    return DynamicTemperatureBodyProperties(
        mass_density=1.0,
        mu=4.0,
        lambda_=4.0,
        theta=4.0,
        zeta=4.0,
        thermal_expansion=thermal_expansion_coeff,
        thermal_conductivity=thermal_conductivity_coeff,
    )


default_obstacle_prop = ObstacleProperties(hardness=100.0, friction=5.0)
default_temp_obstacle_prop = TemperatureObstacleProperties(
    hardness=100.0, friction=5.0, heat=0.01
)

M_RECTANGLE = "pygmsh_rectangle"
M_SPLINE = "pygmsh_spline"
M_CIRCLE = "pygmsh_circle"
M_POLYGON = "pygmsh_polygon"

M_CUBE_3D = "meshzoo_cube_3d"
M_BALL_3D = "meshzoo_ball_3d"
M_POLYGON_3D = "pygmsh_polygon_3d"
M_TWIST_3D = "pygmsh_twist_3d"

o_front = np.array([[[-1.0, 0.0]], [[2.0, 0.0]]])
o_back = np.array([[[1.0, 0.0]], [[-2.0, 0.0]]])
o_slope = np.array([[[-1.0, -2.0]], [[4.0, 0.0]]])
o_side = np.array([[[0.0, 1.0]], [[0.0, -3.0]]])
o_two = np.array([[[-1.0, -2.0], [-1.0, 0.0]], [[2.0, 1.0], [3.0, 0.0]]])

o_3d = np.array([[[-1.0, -1.0, 1.0]], [[2.0, 0.0, 0.0]]])


def f_fall(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_data, time
    force = np.array([2.0, -1.0])
    return force


def f_slide(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_data
    force = np.array([0.0, 0.0])
    if time <= 0.5:
        force = np.array([4.0, 0.0])
    return force


def f_accelerate_fast(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_data, time
    force = np.array([2.0, 0.0])
    return force


def f_accelerate_slow_right(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_data, time
    force = np.array([0.5, 0.0])
    return force


def f_accelerate_slow_left(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_data, time
    force = np.array([-0.5, 0.0])
    return force


def f_stay(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_data, time
    return np.array([0.0, 0.0])


def f_rotate(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = moved_node
    if time <= 0.5:
        y_scaled = initial_node[1] / mesh_data.scale_y
        return y_scaled * np.array([1.5, 0.0])
    return np.array([0.0, 0.0])


def f_rotate_fast(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = moved_node
    if time <= 0.5:
        y_scaled = initial_node[1] / mesh_data.scale_y
        return y_scaled * np.array([3.0, 0.0])
    return np.array([0.0, 0.0])


def f_push_3d(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_data, time
    return np.array([1.0, 1.0, 1.0])


def f_rotate_3d(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_data: MeshProperties,
    time: float,
):
    _ = moved_node, mesh_data
    if time <= 0.5:
        scale = initial_node[1] * initial_node[2]
        return scale * np.array([4.0, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0])


def circle_slope(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        name=f"circle_slope{tag}",
        mesh_data=MeshProperties(
            dimension=2,
            mesh_type=M_CIRCLE,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        body_prop=default_body_prop,
        obstacle_prop=default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_slope,
    )


def spline_right(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        name=f"spline_right{tag}",
        mesh_data=MeshProperties(
            dimension=2,
            mesh_type=M_SPLINE,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        body_prop=default_body_prop,
        obstacle_prop=default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_right,
        obstacles=o_front,
    )


def circle_left(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        f"circle_left{tag}",
        MeshProperties(
            dimension=2,
            mesh_type=M_CIRCLE,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_left,
        obstacles=o_back,
    )


def polygon_left(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        f"polygon_left{tag}",
        MeshProperties(
            dimension=2,
            mesh_type=M_POLYGON,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_left,
        obstacles=o_back * scale,
    )


def polygon_slope(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        f"polygon_slope{tag}",
        MeshProperties(
            dimension=2,
            mesh_type=M_POLYGON,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_slope,
    )


def circle_rotate(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        f"circle_rotate{tag}",
        MeshProperties(
            dimension=2,
            mesh_type=M_CIRCLE,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate,
        obstacles=o_side,
    )


def polygon_rotate(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        f"polygon_rotate{tag}",
        MeshProperties(
            dimension=2,
            mesh_type=M_POLYGON,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate,
        obstacles=o_side,
    )


def polygon_stay(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        f"polygon_stay{tag}",
        MeshProperties(
            dimension=2,
            mesh_type=M_POLYGON,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_stay,
        obstacles=o_side,
    )


def polygon_two(mesh_density, scale, is_adaptive, final_time, tag=""):
    return Scenario(
        f"polygon_two{tag}",
        MeshProperties(
            dimension=2,
            mesh_type=M_POLYGON,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_two,
    )


def get_train_data(**args):
    tag = "_train"
    return [
        polygon_two(**args, tag=tag),
        spline_right(**args, tag=tag),
        circle_left(**args, tag=tag),
        circle_rotate(**args, tag=tag),
        polygon_stay(**args, tag=tag),
    ]


def get_valid_data(**args):
    tag = "_val"
    return [
        polygon_left(**args, tag=tag),
        polygon_rotate(**args, tag=tag),
        circle_slope(**args, tag=tag),
    ]


def all_train(td: TrainingData):
    return get_train_data(
        mesh_density=td.MESH_DENSITY,
        scale=td.TRAIN_SCALE,
        is_adaptive=td.ADAPTIVE_TRAINING_MESH,
        final_time=td.FINAL_TIME,
    )


def all_validation(td: TrainingData):
    return get_valid_data(
        mesh_density=td.MESH_DENSITY,
        scale=td.VALIDATION_SCALE,
        is_adaptive=False,
        final_time=td.FINAL_TIME,
    )


def all_print(td: TrainingData):
    return [
        *get_valid_data(
            mesh_density=td.MESH_DENSITY,
            scale=td.PRINT_SCALE,
            is_adaptive=False,
            final_time=td.FINAL_TIME,
        ),
        *get_train_data(
            mesh_density=td.MESH_DENSITY,
            scale=td.PRINT_SCALE,
            is_adaptive=False,
            final_time=td.FINAL_TIME,
        ),
    ]
