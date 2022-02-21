import math
import os
import resource
import time
from ctypes import ArgumentError
from datetime import datetime
from multiprocessing import Process, Queue

import numpy as np
import pandas
import psutil
import torch
from numba import njit
from tqdm import tqdm

import deep_conmech.common.config as config

# np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_timestamp():
    return int(time.time() * 10000)

RUN_TIMESTEMP = get_timestamp()

CURRENT_TIME = datetime.now().strftime("%d-%H.%M.%S")


# torch.autograd.set_detect_anomaly(True)
print(f"RUNNING USING {device}")


def cuda_launch_blocking():
    print("CUDA_LAUNCH_BLOCKING !!!!")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"






def set_precision(data):
    return data.float()  # float


def to_torch_double(data):
    return torch.tensor(data, dtype=torch.float64)  # .to(device)


def to_torch_long(data):
    return torch.tensor(data, dtype=torch.long)  # .to(device)


def to_np_double(data):
    return data.cpu().detach().numpy().astype(np.float64)

def to_np_long(data):
    return data.cpu().detach().numpy().astype(np.long)


def stack(data):
    return data.T.flatten()


# return data.flatten('F')
# self.F_vector = np.append(self.F[:,0],self.F[:,1])


def stack_column(data):
    return data.T.flatten().reshape(-1, 1)

@njit
def stack_column_numba(data):
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


@njit
def get_random_normal_circle(count, scale):
    result = np.zeros((count, config.DIM))
    for i in range(count):
        alpha = 2 * math.pi * np.random.uniform(low=0, high=1)
        r = np.abs(np.random.normal(loc=0.0, scale=scale * 0.5))
        result[i] = [r * math.cos(alpha), r * math.sin(alpha)]
    return result

@njit
def get_forces_by_function(forces_function, setting, current_time):
    forces = np.zeros((setting.points_number, 2))
    for i in range(setting.points_number):
        initial_point = setting.initial_points[i]
        moved_point = setting.moved_points[i]
        forces[i] = forces_function(initial_point, moved_point, current_time, setting.scale)

    return forces


def norm(data):
    return np.sqrt((data ** 2).sum(-1))[..., np.newaxis] #.reshape(-1,1)

def normalize(data):
    #return np.divide(data, np.linalg.norm(data, axis=-1))
    return data / norm(data)


def euclidean_norm_torch(vector):
    return torch.sqrt(torch.sum(vector ** 2, axis=-1))

@njit
def euclidean_norm(vector):
    return np.sqrt(np.sum(vector ** 2, axis=-1))


def max_norm(data):
    return torch.max(torch.linalg.norm(data, axis=1)) #-1 ?


def rmse_torch(predicted, exact):
    return torch.sqrt(torch.mean(torch.linalg.norm(predicted - exact, axis=-1) ** 2))




def get_tqdm(iterable, desc=None, position=None):
    return tqdm(iterable, desc=desc, position=position, ascii=True)


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

def get_occurances(data):
    return np.array(list(set(data.flatten())))


def get_oriented_tangential_torch(normal):
    tangential = torch.tensor([normal[1], -normal[0]]).to(device)
    return tangential

def rotate_up_torch(old_vectors, up_vector):
    tangential = get_oriented_tangential_torch(up_vector)
    result = torch.vstack((old_vectors @ tangential, old_vectors @ up_vector)).T
    return result


def elementwise_dot_torch(x, y):
    return (x * y).sum(axis=1)
@njit
def elementwise_dot(x, y):
    return (x * y).sum(axis=1)

@njit
def internal_tuple_to_array(tuple, argument):
    return np.array(tuple) if argument.ndim == 1 else np.vstack(tuple).T

    
@njit
def get_oriented_tangential(normal):
    tuple = (normal[...,1], -normal[...,0])
    result = internal_tuple_to_array(tuple, normal)
    return result

@njit
def rotate_up(old_vectors, up_vector):
    tangential = get_oriented_tangential(up_vector)
    tuple = (old_vectors @ tangential, old_vectors @ up_vector)
    result = internal_tuple_to_array(tuple, old_vectors)
    return result


@njit
def calculate_angle(new_up_vector):
    old_up_vector = np.array([0., 1.])
    angle = (2 * (new_up_vector[0] >= 0) - 1) * np.arccos(np.dot(new_up_vector, old_up_vector))
    return angle

@njit
def rotate(vectors, angle):
    s = np.sin(angle)
    c = np.cos(angle)

    rotated_vectors = np.zeros_like(vectors)
    rotated_vectors[:, 0] = vectors[:, 0] * c - vectors[:, 1] * s
    rotated_vectors[:, 1] = vectors[:, 0] * s + vectors[:, 1] * c
    
    return rotated_vectors


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


def print_pandas(data):
    name = f"{data=}".split("=")[0]
    print(f">>> {name} <<<")
    print(pandas.DataFrame(data).round(4))


def get_used_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


def set_memory_limit():
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    # print('Soft limit starts as  :', soft)
    limit = config.TOTAL_MEMORY_LIMIT_GB * (1024 ** 3)  # (b -> kb -> mb -> gb)
    resource.setrlimit(rsrc, (limit, hard))  # limit to one kilobyte
    soft, hard = resource.getrlimit(rsrc)
    # print('Soft limit changed to :', soft)





def skip(time, skip):
    return np.allclose(time % skip, 0.0) or np.allclose(time % skip, skip)


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_folders(path):
    all_folders = path.split("/")
    final_path = ""
    for folder in all_folders:
        final_path += f"{folder}/"
        create_folder(final_path)




def run_processes(function, function_args, num_workers):
    queue = Queue()

    processes = [
        Process(target=lambda *args: queue.put(function(*args)), args=function_args + (num_workers, process_id))
        for process_id in range(num_workers)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    for _ in processes:
        done = queue.get()
        if not done:
            return False

    return True



def run_process(function): #, args
    queue = Queue()

    #wrapper = lambda *args : queue.put(function())
    wrapper = lambda: queue.put(function())

    process = Process(target=wrapper)
    process.start()
    process.join()
    result = queue.get()

    return result
