import functools
from typing import Any, Callable, Optional, Sequence
from unicodedata import name

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import lax
from jax import numpy as jnp
from jax import random
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import flax
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, train):
        x = nn.BatchNorm(use_running_average=not train, name="bn_init")(x)

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))

        x = nn.Dropout(rate=0.1, deterministic=not train)(x)

        x = nn.Dense(features=256)(x)
        # x = nn.Dropout(rate=0.1, deterministic=not self.train)(x)
        # x = nn.BatchNorm(use_running_average=not self.train)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        # x = nn.LayerNorm(dtype=config.dtype)(x)
        return x

    def get_params(self, init_rng):
        params_init_rng, dropout_init_rng = random.split(init_rng, 2)
        rngs_dict = {"params": params_init_rng, "dropout": dropout_init_rng}
        params = self.init(rngs_dict, jnp.ones([1, *mnist_img_size]), train=False)["params"]
        return params


def forward(state, params, data, dropout_rng, train: bool):
    variables = {"params": params, "batch_stats": state.batch_stats}
    rngs = {"dropout": dropout_rng} if train else None
    result, non_trainable_params = state.apply_fn(
        variables, data, rngs=rngs, mutable=["batch_stats"], train=train
    )
    return result, non_trainable_params


def custom_transform(x):
    return np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.0


def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))

    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])

    return imgs, labels


mnist_img_size = (28, 28, 1)
batch_size = 128

train_dataset = MNIST(
    root="datasets/train_mnist", train=True, download=True, transform=custom_transform
)
test_dataset = MNIST(
    root="datasets/test_mnist", train=False, download=True, transform=custom_transform
)

train_loader = DataLoader(
    train_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
)

# optimization - loading the whole dataset into memory
train_images = jnp.array(train_dataset.data)
train_lbls = jnp.array(train_dataset.targets)

# np.expand_dims is to convert shape from (10000, 28, 28) -> (10000, 28, 28, 1)
# We don't have to do this for training images because custom_transform does it for us.
test_images = np.expand_dims(jnp.array(test_dataset.data), axis=3)
test_lbls = jnp.array(test_dataset.targets)


@jax.jit
def train_step(state, imgs, gt_labels, dropout_rng):
    def loss_fn(params):
        logits, new_model_state = forward(
            state=state, params=params, data=imgs, dropout_rng=dropout_rng, train=True
        )
        one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits, new_model_state

    (loss, logits, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])  # this is the whole update now! concise!
    
    metrics = compute_metrics(
        logits=logits, gt_labels=gt_labels
    )  # duplicating loss calculation but it's a bit cleaner
    return state, metrics


@jax.jit
def eval_step(state, imgs, gt_labels):
    logits, _ = state.forward(
        state=state, params=state.params, data=imgs, dropout_rng=None, train=False
    )
    return compute_metrics(logits=logits, gt_labels=gt_labels)


def train_one_epoch(state, dataloader, dropout_rng):
    dropout_rng, new_dropout_rng = random.split(dropout_rng)
    batch_metrics = []
    for cnt, (imgs, labels) in enumerate(dataloader):
        state, metrics = train_step(state, imgs, labels, dropout_rng)
        batch_metrics.append(metrics)

    # Aggregate the metrics
    batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]
    }

    return state, epoch_metrics_np, new_dropout_rng


def evaluate_model(state, test_imgs, test_lbls):
    metrics = eval_step(state, test_imgs, test_lbls)
    metrics = jax.device_get(metrics)
    metrics = jax.tree_map(lambda x: x.item(), metrics)
    return metrics


def create_train_state(learning_rate, init_rng):
    params = CNN().get_params(init_rng)
    optimizer = optax.adam(learning_rate=learning_rate)
    ts = TrainState.create(apply_fn=CNN().apply, params=params, tx=optimizer)
    return ts


def compute_metrics(*, logits, gt_labels):
    one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
    loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


learning_rate = 0.01
num_epochs = 20
batch_size = 32


main_rng = random.PRNGKey(0)
init_rng, dropout_rng = random.split(main_rng, 2)

train_state = create_train_state(learning_rate, init_rng)

for epoch in range(1, num_epochs + 1):
    train_state, train_metrics, dropout_rng = train_one_epoch(
        train_state, train_loader, dropout_rng
    )
    print(
        f"Train epoch: {epoch}, loss: {train_metrics['loss']}, accuracy: {train_metrics['accuracy'] * 100}"
    )

    test_metrics = evaluate_model(train_state, test_images, test_lbls)
    print(
        f"Test epoch: {epoch}, loss: {test_metrics['loss']}, accuracy: {test_metrics['accuracy'] * 100}"
    )
