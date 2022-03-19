import numpy as np
from conmech.dataclass.body_coeff import BodyCoeff
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.obstacle_coeff import ObstacleCoeff
from conmech.dataclass.time_data import TimeData

from deep_conmech.common import config
from deep_conmech.graph.setting.setting_randomized import SettingRandomized


class Scenario:
    def __init__(
        self,
        id,
        mesh_data: MeshData,
        body_coeff: BodyCoeff,
        obstacle_coeff: ObstacleCoeff,
        time_data: TimeData,
        forces_function,
        obstacles,
        is_randomized=None,
    ):
        self.id = id
        self.mesh_data = mesh_data
        self.body_coeff = body_coeff
        self.obstacle_coeff = obstacle_coeff
        self.time_data = time_data
        self.obstacles = obstacles * mesh_data.scale_x
        self.is_randomized = is_randomized
        if isinstance(forces_function, np.ndarray):
            self.forces_function = lambda ip, mp, t, scale_x, scale_y: forces_function
        else:
            self.forces_function = forces_function

    def get_setting(self, randomize, create_in_subprocess=False):
        setting = SettingRandomized(
            mesh_data=self.mesh_data,
            body_coeff=self.body_coeff,
            obstacle_coeff=self.obstacle_coeff,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_randomization(randomize)
        setting.set_obstacles(self.obstacles)
        return setting


####################################

body_coeff = BodyCoeff(mu=4.0, lambda_=4.0, theta=4.0, zeta=4.0)
obstacle_coeff = ObstacleCoeff(hardness=100.0, friction=10.0)

time_data = TimeData(time_step=0.01, final_time=4.0)

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
o_two = np.array([[[-1.0, -2.0], [-1.0, 0.0]], [[3.0, 1.0], [4.0, 0.0]]])


o_3d = np.array([[[-1.0, -1.0, 1.0]], [[2.0, 0.0, 0.0]]])

##################


def f_fall(ip, mp, t, scale_x, scale_y):
    force = np.array([0.2, -0.1])
    return force


def f_slide(ip, mp, t, scale_x, scale_y):
    force = np.array([0.0, 0.0])
    if t <= 0.1:
        force = np.array([0.2, 0.0])
    return force


def f_accelerate_fast(ip, mp, t, scale_x, scale_y):
    force = np.array([0.2, 0.0])
    return force


def f_accelerate_slow_right(ip, mp, t, scale_x, scale_y):
    force = np.array([0.01, 0.0])
    return force


def f_accelerate_slow_left(ip, mp, t, scale_x, scale_y):
    force = np.array([-0.01, 0.0])
    return force


def f_push(ip, mp, t, scale_x, scale_y):
    force = np.array([0.0, 0.0])
    if ip[0] == 0.0:
        p = 5.0 * np.maximum(min[0] - mp[0], 0.0)
        force += p * np.array([1.0, 0.0])
    if t <= 1.0:
        force += np.array([-0.02, 0.0])
    return force


def f_obstacle(ip, mp, t, scale_x, scale_y):
    force = np.array([0.0, 0.0])
    obstacle_x = 1.5
    obstacle_y = 0.7

    if (
        ip[0] == scale_x
        and ip[1] < obstacle_y
        and mp[0] > obstacle_x
        and mp[1] < obstacle_y
    ):
        p = 2.0 * np.maximum(mp[0] - obstacle_x, 0.0)
        force += p * np.array([-1.0, 0.0])
    if t <= 1.0:
        force += np.array([0.02, 0.0])
    return force


def f_stay(ip, mp, t, scale_x, scale_y):
    return np.array([0.0, 0.0])


# def get_f_rotate(t_cutoff = 0.1, force_cutoff=0.5):
def f_rotate(ip, mp, t, scale_x, scale_y):
    t_cutoff = 0.5
    force_cutoff = 10.0
    if t <= t_cutoff:
        y_scaled = ip[1] / scale_y
        # y_scaled = 2*y_scaled - 1.
        return y_scaled * np.array([force_cutoff, 0.0])
    return np.array([0.0, 0.0])


def f_random(ip, mp, t, scale_x, scale_y):
    scale = config.FORCES_RANDOM_SCALE
    force = np.random.uniform(low=-scale, high=scale, size=2)
    return force


####################################


def f_push_3d(ip, mp, t, scale_x, scale_y):
    return np.array([0.05, 0.05, 0.05])
    # return np.repeat(np.array([f0]), nodes_count, axis=0)


def f_rotate_3d(ip, mp, t, scale_x, scale_y):
    if t <= 0.5:
        scale = ip[1] * ip[2]
        return scale * np.array([0.1, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0])


####################################



def circle_slope(scale, is_adaptive):
    return Scenario(
        id="circle_slope",
        mesh_data=MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff=body_coeff,
        obstacle_coeff=obstacle_coeff,
        time_data=time_data,
        forces_function=f_accelerate_slow_right,
        obstacles=o_slope,
    )


def spline_right(scale, is_adaptive):
    return Scenario(
        "spline_right",
        MeshData(
            dimension=2,
            mesh_type=m_spline,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff,
        obstacle_coeff,
        time_data,
        f_accelerate_slow_right,
        o_front
    )


def circle_left(scale, is_adaptive):
    return Scenario(
        "circle_left",
        MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff,
        obstacle_coeff,
        time_data,
        f_accelerate_slow_left,
        o_back * scale
    )


def polygon_left(scale, is_adaptive):
    return Scenario(
        "polygon_left",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff,
        obstacle_coeff,
        time_data,
        f_accelerate_slow_left,
        o_back * scale,
    )


def polygon_slope(scale, is_adaptive):
    return Scenario(
        "polygon_slope",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff,
        obstacle_coeff,
        time_data,
        f_slide,
        o_slope
    )


def circle_rotate(scale, is_adaptive):
    return Scenario(
        "circle_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff,
        obstacle_coeff,
        time_data,
        f_rotate,
        o_side
    )


def polygon_rotate(scale, is_adaptive):
    return Scenario(
        "polygon_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff,
        obstacle_coeff,
        time_data,
        f_rotate,
        o_side
    )


def polygon_stay(scale, is_adaptive):
    return Scenario(
        "polygon_stay",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff,
        obstacle_coeff,
        time_data,
        f_stay,
        o_side
    )


def polygon_two(scale, is_adaptive):
    return Scenario(
        "polygon_two",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[config.MESH_DENSITY],
            is_adaptive=is_adaptive,
        ),
        body_coeff,
        obstacle_coeff,
        time_data,
        f_slide,
        o_two
    )


def get_data(scale, is_adaptive):
    return [
        polygon_two(scale, is_adaptive),
        circle_slope(scale, is_adaptive),
        spline_right(scale, is_adaptive),
        circle_left(scale, is_adaptive),
        polygon_left(scale, is_adaptive),
        circle_rotate(scale, is_adaptive),
        polygon_rotate(scale, is_adaptive),
        polygon_stay(scale, is_adaptive),
    ]


all_train = get_data(
    scale=config.TRAIN_SCALE, is_adaptive=config.ADAPTIVE_TRAINING_MESH
)

all_validation = get_data(scale=config.VALIDATION_SCALE, is_adaptive=False)

all_print = [
    # *get_data(scale=config.PRINT_SCALE, is_adaptive=False),
    # *get_data(scale=config.VALIDATION_SCALE, is_adaptive=False),
    polygon_two(scale=config.PRINT_SCALE, is_adaptive=False),
    circle_slope(scale=config.PRINT_SCALE, is_adaptive=False),
    circle_left(scale=config.PRINT_SCALE, is_adaptive=False),
    circle_rotate(scale=config.PRINT_SCALE, is_adaptive=False),
]
