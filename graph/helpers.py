import math
import time
from ctypes import ArgumentError
from datetime import datetime

import numba
import numpy as np
import torch
import torch.nn as nn
from numba import cuda, jit, njit, prange
from tqdm import tqdm

import config

# np.random.seed(42)

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.autograd.set_detect_anomaly(True)
print(f"RUNNING USING {device}")


def torch_zeros_float(shape):
    return torch.zeros(shape, dtype=torch.float)  # .to(device)
    # from_numpy - shared memory


def to_torch_float(data):
    return torch.tensor(data, dtype=torch.float)  # .to(device)


def to_torch_double(data):
    return torch.tensor(data, dtype=torch.float64)  # .to(device)


def to_torch_long(data):
    return torch.tensor(data, dtype=torch.long)  # .to(device)


def to_np(data):
    return data.cpu().detach().numpy()


def stack(data):
    return data.T.flatten()


# return data.flatten('F')
# self.F_vector = np.append(self.F[:,0],self.F[:,1])


def stack_column(data):
    return data.T.flatten().reshape(-1, 1)


def unstack(data):
    return data.reshape(-1, config.DIM, order="F")


@njit
def get_point_index(point, points):
    for i in range(len(points)):
        if np.sum(np.abs(point - points[i])) < 0.0001:
            return i
    raise ArgumentError


# @njit
def get_random_normal(count, scale):
    # noise = np.random.uniform(
    #    low=-scale, high=scale, size=shape
    # )

    noise = np.random.normal(loc=0.0, scale=scale * 0.5, size=[count, config.DIM])
    return noise


def get_random_normal_circle(count, scale):
    result = np.zeros((count, config.DIM))
    for i in range(count):
        alpha = 2 * math.pi * np.random.uniform(low=0, high=1)
        r = np.abs(np.random.normal(loc=0.0, scale=scale * 0.5))
        result[i] = [r * math.cos(alpha), r * math.sin(alpha)]
    return result



def get_forces_by_function(forces_function, setting, time):
    forces = np.zeros((setting.points_number, 2))
    for i in range(setting.points_number):
        initial_point = setting.initial_points[i]
        moved_point = setting.moved_points[i]
        forces[i] = forces_function(initial_point, moved_point, time, setting.corners)

    return forces


def max_norm(data):
    return torch.max(torch.linalg.norm(data, axis=1))


def avg_norm(data):
    return torch.mean(torch.linalg.norm(data, axis=1))


def get_timestamp():
    return int(time.time() * 10000)


RUN_TIMESTEMP = get_timestamp()

CURRENT_TIME = datetime.now().strftime("%d-%H.%M.%S")


def get_tqdm(iterable, desc):
    return tqdm(iterable, desc=desc, ascii=True,)


@njit
def len_x(corners):
    return corners[2] - corners[0]

@njit
def len_y(corners):
    return corners[3] - corners[1]

@njit
def min(corners):
    return [corners[0], corners[1]]

@njit
def max(corners):
    return [corners[2], corners[3]]



class MaxData:
    def __init__(self, description, episode_steps):
        self.description = description
        self.max_forces = torch.zeros(episode_steps)
        self.max_u = torch.zeros(episode_steps)
        self.max_v = torch.zeros(episode_steps)

    def set(self, setting, i):
        self.max_forces[i] = max_norm(setting.normalized_forces_torch)
        self.max_u[i] = max_norm(setting.normalized_u_old_torch)
        self.max_v[i] = max_norm(setting.normalized_v_old_torch)

    def print(self):
        print(
            f"max norms -{self.description} | f: {float(torch.max(self.max_forces)):.4f} | u: {float(torch.max(self.max_u)):.4f} | v: {float(torch.max(self.max_v))}"
        )


def get_base_density(mesh_size, corners):
    return len_x(corners) / mesh_size


def mesh_corner_data(base_density):
    scale = base_density * 0.3 # 0.4
    corner_data = np.random.uniform(low=-scale, high=scale, size=4)
    return corner_data

@njit
def get_adaptive_mesh_density(x, y, base_density, corner_data):
    correction_left = x * corner_data[0] + (1 - x) * (corner_data[1] - corner_data[0])
    correction_right = x * corner_data[2] + (1 - x) * (corner_data[3] - corner_data[2])
    correction = y * correction_left + (1 - y) * (correction_right - correction_left)
    mesh_density = base_density + correction
    return mesh_density
    #z = np.sin(np.sqrt(x**2 + y**2))
    #z = 2*(6.0e-2) + 2*(2.0e-1) * ((x+0.5) ** 2 + y ** 2)
    #return z 
