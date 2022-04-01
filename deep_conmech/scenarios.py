from typing import Callable, Optional, Union

import numpy as np

from conmech.dataclass.body_properties import DynamicBodyProperties, \
    DynamicTemperatureBodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.obstacle_properties import ObstacleProperties, TemperatureObstacleProperties
from conmech.dataclass.schedule import Schedule
from conmech.helpers import cmh
from conmech.helpers.config import Config
from deep_conmech.common.training_data import TrainingData
from deep_conmech.simulator.setting.setting_iterable import SettingIterable
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature
from deep_conmech.simulator.solver import Solver


class Scenario:
    def __init__(
            self,
            id: str,
            mesh_data: MeshData,
            body_prop: DynamicBodyProperties,
            obstacle_prop: ObstacleProperties,
            schedule: Schedule,
            forces_function: Union[Callable[..., np.ndarray], np.ndarray],
            obstacles: Optional[np.ndarray],
    ):
        self.id = id
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
            desc=f"{desc} {self.id}",
        )

    def get_solve_function(self):
        return Solver.solve

    def get_setting(
            self, normalize_by_rotation=True, randomize=False, create_in_subprocess: bool = False
    ) -> SettingIterable:
        setting = SettingIterable(
            mesh_data=self.mesh_data,
            body_prop=self.body_prop,
            obstacle_prop=self.obstacle_prop,
            schedule=self.schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_obstacles(self.obstacles)
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
            id: str,
            mesh_data: MeshData,
            body_prop: DynamicTemperatureBodyProperties,
            obstacle_prop: TemperatureObstacleProperties,
            schedule: Schedule,
            forces_function: Union[Callable, np.ndarray],
            obstacles: Optional[np.ndarray],
            heat_function: Union[Callable, np.ndarray],
    ):
        super().__init__(
            id=id,
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

    def get_solve_function(self):
        return Solver.solve_with_temperature

    def get_setting(
            self, normalize_by_rotation=True, randomize=False, create_in_subprocess: bool = False
    ) -> SettingTemperature:
        setting = SettingTemperature(
            mesh_data=self.mesh_data,
            body_prop=self.body_prop,
            obstacle_prop=self.obstacle_prop,
            schedule=self.schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_obstacles(self.obstacles)
        return setting


default_schedule = Schedule(time_step=0.01, final_time=4.0)

###

default_body_prop = DynamicBodyProperties(
    mu=4.0, lambda_=4.0, theta=4.0, zeta=4.0, mass_density=1.0
)
# body_prop = DynamicBodyProperties(mu=0.01, lambda_=0.01, theta=0.01, zeta=0.01, mass_density=0.01)

default_C_coeff = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
default_K_coeff = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
default_temp_body_prop = DynamicTemperatureBodyProperties(
    mass_density=1.0,
    mu=4.0,
    lambda_=4.0,
    theta=4.0,
    zeta=4.0,
    C_coeff=default_C_coeff,
    K_coeff=default_K_coeff,
)


def get_temp_body_prop(C_coeff, K_coeff):
    return DynamicTemperatureBodyProperties(
        mass_density=1.0,
        mu=4.0,
        lambda_=4.0,
        theta=4.0,
        zeta=4.0,
        C_coeff=C_coeff,
        K_coeff=K_coeff,
    )

###


default_obstacle_prop = ObstacleProperties(hardness=100.0, friction=5.0)
default_temp_obstacle_prop = TemperatureObstacleProperties(hardness=100.0, friction=5.0, heat=0.01)

m_rectangle = "pygmsh_rectangle"
m_spline = "pygmsh_spline"
m_circle = "pygmsh_circle"
m_polygon = "pygmsh_polygon"

m_cube_3d = "meshzoo_cube_3d"
m_ball_3d = "meshzoo_ball_3d"
m_polygon_3d = "pygmsh_polygon_3d"
m_twist_3d = "pygmsh_twist_3d"

o_front = np.array([[[-1.0, 0.0]], [[2.0, 0.0]]])
o_back = np.array([[[1.0, 0.0]], [[-2.0, 0.0]]])
o_slope = np.array([[[-1.0, -2.0]], [[4.0, 0.0]]])
o_side = np.array([[[0.0, 1.0]], [[0.0, -3.0]]])
o_two = np.array([[[-1.0, -2.0], [-1.0, 0.0]], [[2.0, 1.0], [3.0, 0.0]]])

o_3d = np.array([[[-1.0, -1.0, 1.0]], [[2.0, 0.0, 0.0]]])


def f_fall(ip, mp, md, t):
    force = np.array([2.0, -1.0])
    return force


def f_slide(ip, mp, md, t):
    force = np.array([0.0, 0.0])
    if t <= 0.5:
        force = np.array([4.0, 0.0])
    return force


def f_accelerate_fast(ip, mp, md, t):
    force = np.array([2.0, 0.0])
    return force


def f_accelerate_slow_right(ip, mp, md, t):
    force = np.array([0.5, 0.0])
    return force


def f_accelerate_slow_left(ip, mp, md, t):
    force = np.array([-0.5, 0.0])
    return force


def f_stay(ip, mp, md, t):
    return np.array([0.0, 0.0])


def f_rotate(ip, mp, md, t):
    if t <= 0.5:
        y_scaled = ip[1] / md.scale_y
        return y_scaled * np.array([1.5, 0.0])
    return np.array([0.0, 0.0])


def f_rotate_fast(ip, mp, md, t):
    if t <= 0.5:
        y_scaled = ip[1] / md.scale_y
        return y_scaled * np.array([3.0, 0.0])
    return np.array([0.0, 0.0])


def f_push_3d(ip, mp, md, t):
    return np.array([1.0, 1.0, 1.0])


def f_rotate_3d(ip, mp, md, t):
    if t <= 0.5:
        scale = ip[1] * ip[2]
        return scale * np.array([4.0, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0])


def circle_slope(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        id="circle_slope",
        mesh_data=MeshData(
            dimension=2,
            mesh_type=m_circle,
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


def spline_right(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        id="spline_right",
        mesh_data=MeshData(
            dimension=2,
            mesh_type=m_spline,
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


def circle_left(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        "circle_left",
        MeshData(
            dimension=2,
            mesh_type=m_circle,
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


def polygon_left(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        "polygon_left",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
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


def polygon_slope(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        "polygon_slope",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
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


def circle_rotate(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        "circle_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_circle,
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


def polygon_rotate(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        "polygon_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
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


def polygon_stay(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        "polygon_stay",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
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


def polygon_two(mesh_density, scale, is_adaptive, final_time):
    return Scenario(
        "polygon_two",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
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


def get_data(**args):
    return [
        polygon_rotate(**args),
        polygon_two(**args),
        circle_slope(**args),
        spline_right(**args),
        circle_left(**args),
        polygon_left(**args),
        circle_rotate(**args),
        polygon_rotate(**args),
        polygon_rotate(**args),
        polygon_stay(**args),
    ]


def all_train(td: TrainingData):
    return get_data(
        mesh_density=td.MESH_DENSITY,
        scale=td.TRAIN_SCALE,
        is_adaptive=td.ADAPTIVE_TRAINING_MESH,
        final_time=td.FINAL_TIME,
    )


def all_validation(td: TrainingData):
    return get_data(
        mesh_density=td.MESH_DENSITY,
        scale=td.VALIDATION_SCALE,
        is_adaptive=False,
        final_time=td.FINAL_TIME,
    )


def print_args(td: TrainingData):
    return dict(
        mesh_density=td.MESH_DENSITY,
        scale=td.PRINT_SCALE,
        is_adaptive=False,
        final_time=td.FINAL_TIME,
    )


def all_print(td: TrainingData):
    return [
        *get_data(
            mesh_density=td.MESH_DENSITY,
            scale=td.PRINT_SCALE,
            is_adaptive=False,
            final_time=td.FINAL_TIME,
        ),
        # *get_data(scale=config.VALIDATION_SCALE, is_adaptive=False, final_time=config.FINAL_TIME),
        # polygon_two(**print_args),
    ]
