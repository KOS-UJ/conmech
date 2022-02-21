import numpy as np

import config
import helpers


def f_slide(ip, mp, t, corners):
    force = np.array([0.0, 0.0])
    if t <= 0.1:
        force = np.array([0.2, 0.0])
    return force

def f_accelerate(ip, mp, t, cornersx):
    force = np.array([0.3, 0.0])
    return force


def f_push(ip, mp, t, corners):
    min = helpers.min(corners)
    force = np.array([0.0, 0.0])
    if ip[0] == min[0]:
        p = 5.0 * np.maximum(min[0] - mp[0], 0.0)
        force += p * np.array([1.0, 0.0])
    if t <= 1.0:
        force += np.array([-0.02, 0.0])
    return force


def f_obstacle(ip, mp, t, corners):
    min = helpers.min(corners)
    max = helpers.max(corners)
    force = np.array([0.0, 0.0])
    obstacle_x = 1.5
    obstacle_y = 0.7

    if (
        ip[0] == max[0]
        and ip[1] < obstacle_y
        and mp[0] > obstacle_x
        and mp[1] < obstacle_y
    ):
        p = 2.0 * np.maximum(mp[0] - min[0] - obstacle_x, 0.0)
        force += p * np.array([-1.0, 0.0])
    if t <= 1.0:
        force += np.array([0.02, 0.0])
    return force


def f_tug_and_rotate(ip, mp, t, corners):
    min = helpers.min(corners)
    max = helpers.max(corners)
    force = np.array([0.0, 0.0])
    rotate_scalar = 0.4
    tug_scalar = 0.8

    if t <= 0.1:
        y_scaled = (ip[1] - min[1]) / helpers.len_y(min, max)
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


# def get_f_rotate(t_cutoff = 0.1, force_cutoff=0.5):
def f_rotate(ip, mp, t, corners):
    min = helpers.min(corners)
    t_cutoff = 0.1
    force_cutoff = 0.5
    if t <= t_cutoff:
        y_scaled = (ip[1] - min[1]) / helpers.len_y(corners)
        # y_scaled = 2*y_scaled - 1.
        return y_scaled * np.array([force_cutoff, 0.0])
    return np.array([0.0, 0.0])


# return f_rotate


def reverse(f):
    return lambda ip, mp, t, corners: -f(ip, mp, t, corners)


def f_random(ip, mp, t, corners):
    scale = config.FORCES_RANDOM_SCALE
    force = np.random.uniform(low=-scale, high=scale, size=2)
    return force


def f_drag(ip, mp, t, corners):
    max = helpers.max(corners)
    if t <= 0.1 and ip[1] == max[1]:
        return np.array([1.0, 0.0])
    return np.array([0.0, 0.0])
