'''
torch helpers
'''
import numpy as np
import torch

from deep_conmech.common.training_config import TrainingConfig


def device(training_config: TrainingConfig):
    return torch.device(training_config.DEVICE)


def get_device_id():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_precision(data):
    return data.float()


def to_torch_double(data):
    return torch.tensor(data, dtype=torch.float64)


def to_torch_long(data):
    return torch.tensor(data, dtype=torch.long)


def to_np_double(data):
    return data.cpu().detach().numpy().astype(np.float64)


def to_np_long(data):
    return data.cpu().detach().numpy().astype(np.long)


def get_contiguous_torch(data):
    return to_torch_long(data).t().contiguous()


def append_euclidean_norm(data):
    return torch.hstack((data, torch.linalg.norm(data, keepdim=True, dim=1)))


def euclidean_norm_torch(vector):
    return torch.sqrt(torch.sum(vector ** 2, axis=-1))


def max_norm(data):
    return torch.max(torch.linalg.norm(data, axis=1))  # -1 ?


def rmse_torch(predicted, exact):
    return torch.sqrt(torch.mean(torch.linalg.norm(predicted - exact, axis=-1) ** 2))


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
