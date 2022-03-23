from typing import Callable, Union

import numpy as np
from conmech.dataclass.body_properties import BodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.obstacle_properties import ObstacleProperties
from conmech.dataclass.schedule import Schedule
from conmech.helpers import cmh

from deep_conmech.common import config


class Scenario:
    def __init__(
        self,
        id: str,
        mesh_data: MeshData,
        body_prop: BodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        forces_function: Union[Callable, np.ndarray],
        obstacles: np.ndarray,
    ):
        self.id = id
        self.mesh_data = mesh_data
        self.body_prop = body_prop
        self.obstacle_prop = obstacle_prop
        self.schedule = schedule
        self.obstacles = obstacles * mesh_data.scale_x
        if isinstance(forces_function, np.ndarray):
            self.forces_function = lambda ip, mp, t, scale_x, scale_y: forces_function
        else:
            self.forces_function = forces_function

    def get_tqdm(self, description):
        return cmh.get_tqdm(
            range(self.schedule.episode_steps),
            f"{description} {self.id} scale_{self.mesh_data.scale_x}",
        )

    @property
    def dimension(self):
        return self.mesh_data.dimension

    @property
    def time_step(self):
        return self.schedule.time_step
    @property
    def final_time(self):
        return self.schedule.final_time


####################################

body_prop = BodyProperties(mu=4.0, lambda_=4.0, theta=4.0, zeta=4.0, mass_density=1.0)
# body_prop = BodyProperties(mu=0.01, lambda_=0.01, theta=0.01, zeta=0.01, mass_density=0.01)
obstacle_prop = ObstacleProperties(hardness=100.0, friction=5.0)

schedule = Schedule(time_step=0.01, final_time=4.0)

####################################

m_rectangle = "pygmsh_rectangle"
m_spline = "pygmsh_spline"
m_circle = "pygmsh_circle"
m_polygon = "pygmsh_polygon"

m_cube_3d = "meshzoo_cube_3d"
m_ball_3d = "meshzoo_ball_3d"
m_polygon_3d = "pygmsh_polygon_3d"

####################################

o_front = np.array([[[-1.0, 0.0]], [[2.0, 0.0]]])
o_back = np.array([[[1.0, 0.0]], [[-2.0, 0.0]]])
o_slope = np.array([[[-1.0, -2.0]], [[4.0, 0.0]]])
o_side = np.array([[[0.0, 1.0]], [[0.0, -3.0]]])
o_two = np.array([[[-1.0, -2.0], [-1.0, 0.0]], [[2.0, 1.0], [3.0, 0.0]]])


o_3d = np.array([[[-1.0, -1.0, 1.0]], [[2.0, 0.0, 0.0]]])

##################


def f_fall(ip, mp, t, scale_x, scale_y):
    force = np.array([2.0, -1.0])
    return force


def f_slide(ip, mp, t, scale_x, scale_y):
    force = np.array([0.0, 0.0])
    if t <= 0.5:
        force = np.array([4.0, 0.0])
    return force


def f_accelerate_fast(ip, mp, t, scale_x, scale_y):
    force = np.array([2.0, 0.0])
    return force


def f_accelerate_slow_right(ip, mp, t, scale_x, scale_y):
    force = np.array([0.5, 0.0])
    return force


def f_accelerate_slow_left(ip, mp, t, scale_x, scale_y):
    force = np.array([-0.5, 0.0])
    return force


def f_stay(ip, mp, t, scale_x, scale_y):
    return np.array([0.0, 0.0])


def f_rotate(ip, mp, t, scale_x, scale_y):
    if t <= 0.5:
        y_scaled = ip[1] / scale_y
        # y_scaled = 2*y_scaled - 1.
        return y_scaled * np.array([1.5, 0.0])
    return np.array([0.0, 0.0])


def f_random(ip, mp, t, scale_x, scale_y):
    scale = config.FORCES_RANDOM_SCALE
    force = np.random.uniform(low=-scale, high=scale, size=2)
    return force


####################################


def f_push_3d(ip, mp, t, scale_x, scale_y):
    return np.array([1.0, 1.0, 1.0])


def f_rotate_3d(ip, mp, t, scale_x, scale_y):
    if t <= 0.5:
        scale = ip[1] * ip[2]
        return scale * np.array([4.0, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0])


####################################


def circle_slope(scale, is_adaptive, final_time):
    return Scenario(
        id="circle_slope",
        mesh_data=MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop=body_prop,
        obstacle_prop=obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_slope,
    )


def spline_right(scale, is_adaptive, final_time):
    return Scenario(
        id="spline_right",
        mesh_data=MeshData(
            dimension=2,
            mesh_type=m_spline,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop=body_prop,
        obstacle_prop=obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_right,
        obstacles=o_front,
    )


def circle_left(scale, is_adaptive, final_time):
    return Scenario(
        "circle_left",
        MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop,
        obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_left,
        obstacles=o_back,
    )


def polygon_left(scale, is_adaptive, final_time):
    return Scenario(
        "polygon_left",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop,
        obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_left,
        obstacles=o_back * scale,
    )


def polygon_slope(scale, is_adaptive, final_time):
    return Scenario(
        "polygon_slope",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop,
        obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_slope,
    )


def circle_rotate(scale, is_adaptive, final_time):
    return Scenario(
        "circle_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop,
        obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate,
        obstacles=o_side,
    )


def polygon_rotate(scale, is_adaptive, final_time):
    return Scenario(
        "polygon_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop,
        obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate,
        obstacles=o_side,
    )


def polygon_stay(scale, is_adaptive, final_time):
    return Scenario(
        "polygon_stay",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop,
        obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_stay,
        obstacles=o_side,
    )


def polygon_two(scale, is_adaptive, final_time):
    return Scenario(
        "polygon_two",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_prop,
        obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_two,
    )


#########################


def get_data(scale, is_adaptive, final_time):
    return [
        polygon_rotate(scale, is_adaptive, final_time),
        # polygon_two(scale, is_adaptive, final_time),
        circle_slope(scale, is_adaptive, final_time),
        spline_right(scale, is_adaptive, final_time),
        circle_left(scale, is_adaptive, final_time),
        # polygon_left(scale, is_adaptive, final_time),
        # circle_rotate(scale, is_adaptive, final_time),
        # polygon_rotate(scale, is_adaptive, final_time),
        # polygon_stay(scale, is_adaptive, final_time),
    ]


all_train = get_data(
    scale=config.TRAIN_SCALE,
    is_adaptive=config.ADAPTIVE_TRAINING_MESH,
    final_time=config.FINAL_TIME,
)

all_validation = get_data(
    scale=config.VALIDATION_SCALE, is_adaptive=False, final_time=config.FINAL_TIME,
)

print_args = dict(
    scale=config.PRINT_SCALE, is_adaptive=False, final_time=config.FINAL_TIME
)
all_print = [
    *get_data(
        scale=config.PRINT_SCALE, is_adaptive=False, final_time=config.FINAL_TIME
    ),
    # *get_data(scale=config.VALIDATION_SCALE, is_adaptive=False, final_time=config.FINAL_TIME),
    # polygon_two(**print_args),
]
