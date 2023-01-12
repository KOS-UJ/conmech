import jax
import jax.numpy as jnp
import npx
import numpy as np
import scipy
import torch
from sklearn.decomposition import PCA

from conmech.helpers import cmh, jxh, nph
from deep_conmech.data import base_dataset
from deep_conmech.helpers import thh

def run(dataset):
    dataloader = base_dataset.get_train_dataloader(dataset)

    data = []
    count = 110
    for _ in range(count):
        sample = next(iter(dataloader))
        u = sample[0][0][1].pos
        # u = sample[0][1].normalized_exact_acceleration
        # u = sample[0][1].reduced_normalized_lifted_acceleration
        u_stack = nph.stack_column(u)
        data.append(u_stack)

    result_torch = torch.cat(data).reshape(count, -1)
    result = thh.convert_tensor_to_jax(result_torch)
    projection_mean = result.mean(axis=0)
    result = result - projection_mean
    # result.mean(axis=0) - column wise mean == 0

    val = jax.numpy.linalg.svd(result, full_matrices=False)

    K = 8
    projection = val[2][:K].T
    # (val[0] @ jnp.diag(val[1]) @ val[2])

    u_m = thh.convert_tensor_to_jax(u_stack) - projection_mean.reshape(-1, 1)
    l = projection.T @ u_m

    u_m2 = projection @ l
    u2 = u_m2 + projection_mean.reshape(-1, 1)

    return 0
