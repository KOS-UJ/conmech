import numpy as np
import numba
from deep_conmech.common import config


o_front = np.array([[[-1.0, 0.0]], [[2.0, 0.0]]])
o_back = np.array([[[1.0, 0.0]], [[-2.0, 0.0]]])
o_slope = np.array([[[-1.0, -2.0]], [[4.0, 0.0]]])
o_side = np.array([[[0.0, 1.0]], [[0.0, -3.0]]])

o_two = np.array([[[-1.0, -2.0], [-1.0, 0.0]], [[3.0, 1.0], [4.0, 0.0]]])

##################


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


def f_tug_and_rotate(ip, mp, t, scale_x, scale_y):
    min = basic_helpers.min(corners)
    max = basic_helpers.max(corners)
    force = np.array([0.0, 0.0])
    rotate_scalar = 0.4
    tug_scalar = 0.8

    if t <= 0.1:
        y_scaled = (ip[1] - min[1]) / basic_helpers.len_y(min, max)
        force += y_scaled * np.array([rotate_scalar, 0.0])

    if t <= 1.0:
        if ip[0] == min[0]:
            force += np.array([-tug_scalar, 0.0])
        if ip[0] == max[0]:
            force += np.array([tug_scalar, 0.0])
        if ip[1] == max[1]:
            force += np.array([0.0, tug_scalar])
        if ip[1] == min[1]:
            force += np.array([0.0, -tug_scalar])
    return force


def f_stay(ip, mp, t, scale_x, scale_y):
    return np.array([0.0, 0.0])


# def get_f_rotate(t_cutoff = 0.1, force_cutoff=0.5):
def f_rotate(ip, mp, t, scale_x, scale_y):
    t_cutoff = 0.1
    force_cutoff = 0.5
    if t <= t_cutoff:
        y_scaled = ip[1] / scale_y
        # y_scaled = 2*y_scaled - 1.
        return y_scaled * np.array([force_cutoff, 0.0])
    return np.array([0.0, 0.0])


# return f_rotate


def reverse(f):
    return lambda ip, mp, t, scale_x, scale_y: numba.njit(-f(ip, mp, t, scale_x, scale_y))


def f_random(ip, mp, t, scale_x, scale_y):
    scale = config.FORCES_RANDOM_SCALE
    force = np.random.uniform(low=-scale, high=scale, size=2)
    return force


def f_drag(ip, mp, t, scale_x, scale_y):
    max = basic_helpers.max(corners)
    if t <= 0.1 and ip[1] == max[1]:
        return np.array([1.0, 0.0])
    return np.array([0.0, 0.0])


####################################

m_rectangle = "pygmsh_rectangle"
m_circle = "pygmsh_circle"
m_polygon = "pygmsh_polygon"
m_cross = "cross"

####################################


class Scenario:
    def __init__(self, id, mesh_type, mesh_density, scale, forces_function, obstacles, is_adaptive, duration=None, is_randomized=None):
        self.id = id
        self.mesh_type = mesh_type
        self.mesh_density = mesh_density
        self.scale = scale
        self.forces_function = forces_function
        self.obstacles = obstacles
        self.is_adaptive = is_adaptive
        self.duration = duration
        self.is_randomized = is_randomized


def circle_slope(scale, is_adaptive):
    return Scenario(
    "circle_slope",
    m_circle,
    config.MESH_DENSITY,
    scale,
    f_accelerate_slow_right,
    o_slope * scale,
    is_adaptive
)
def circle_right(scale, is_adaptive):
    return Scenario(
    "circle_right",
    m_circle,
    config.MESH_DENSITY,
    scale,
    f_accelerate_slow_right,
    o_front * scale,
    is_adaptive
)
def circle_left(scale, is_adaptive):
    return Scenario(
    "circle_left",
    m_circle,
    config.MESH_DENSITY,
    scale,
    f_accelerate_slow_left,
    o_back * scale,
    is_adaptive
)
def polygon_slope(scale, is_adaptive):
    return Scenario(
    "polygon_slope",
    m_polygon,
    config.MESH_DENSITY,
    scale,
    f_slide,
    o_slope * scale,
    is_adaptive
)
def circle_rotate(scale, is_adaptive):
    return Scenario(
    "circle_rotate",
    m_circle,
    config.MESH_DENSITY,
    scale,
    f_rotate,
    o_side * scale,
    is_adaptive
)

def polygon_stay(scale, is_adaptive):
    return Scenario(
    "polygon_stay",
    m_polygon,
    config.MESH_DENSITY,
    scale,
    f_stay,
    o_side * scale,
    is_adaptive
)

def polygon_two(scale, is_adaptive):
    return Scenario(
    "polygon_two",
    m_polygon,
    config.MESH_DENSITY,
    scale,
    f_slide,
    o_two * scale,
    is_adaptive
)

def cross_slope(scale, is_adaptive):
    return Scenario(
    "cross_slope",
    m_cross,
    config.MESH_DENSITY,
    scale,
    f_slide,
    o_side * scale,
    is_adaptive
)


all_train = [
    #polygon_two(scale=config.TRAIN_SCALE, is_adaptive=True),
    circle_slope(scale=config.TRAIN_SCALE, is_adaptive=True),
    circle_right(scale=config.TRAIN_SCALE, is_adaptive=True),
    #circle_left(scale=config.TRAIN_SCALE, is_adaptive=True),
    circle_rotate(scale=config.TRAIN_SCALE, is_adaptive=True),
    polygon_stay(scale=config.TRAIN_SCALE, is_adaptive=True)
]

all_print = [
    polygon_two(scale=config.PRINT_SCALE, is_adaptive=False),
    #circle_slope(scale=config.PRINT_SCALE, is_adaptive=False),
    #circle_right(scale=config.PRINT_SCALE, is_adaptive=False),
    circle_left(scale=config.PRINT_SCALE, is_adaptive=False),
    circle_rotate(scale=config.PRINT_SCALE, is_adaptive=False),
    polygon_stay(scale=config.PRINT_SCALE, is_adaptive=False)
]
