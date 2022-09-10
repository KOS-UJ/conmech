from typing import List, Optional

import jax.numpy as jnp
import numpy as np
import torch

from conmech.helpers import nph
from conmech.scene.body_forces import energy_lhs, energy_vector_lhs
from conmech.scene.scene import (
    EnergyObstacleArguments,
    energy_obstacle_jax,
    energy_obstacle_torch,
)
from deep_conmech.data.data_classes import EnergyObstacleArgumentsTorch
from deep_conmech.graph.loss_raport import LossRaport
from deep_conmech.helpers import thh


def clean_acceleration(cleaned_a, a_correction):
    return cleaned_a if (a_correction is None) else (cleaned_a - a_correction)


def get_mean_loss(acceleration, forces, mass_density, boundary_integral):
    # F = m * a
    return (boundary_integral == 0) * (
        torch.norm(torch.mean(forces, axis=0) - torch.mean(mass_density * acceleration, axis=0))
        ** 2
    )


def get_acceleration_vector(acceleration, graph_sizes_base):
    predicted_acceleration_split = acceleration.split(graph_sizes_base)
    acceleration_vector = torch.vstack(tuple(map(nph.stack_column, predicted_acceleration_split)))
    return acceleration_vector


def loss_normalized_obstacle_scatter(
    acceleration: torch.Tensor,
    # energy_args: EnergyObstacleArguments,
    graph_sizes_base: List[int],
    exact_acceleration: torch.Tensor,
    linear_acceleration: Optional[torch.Tensor],
):
    num_graphs = len(graph_sizes_base)
    main_loss = thh.root_mean_square_error_torch(acceleration, exact_acceleration)

    loss_raport = LossRaport(
        main=main_loss.item(),
        inner_energy=0,
        energy=0,
        boundary_integral=0,
        mean=0,  # thh.root_mean_square_error_torch(linear_acceleration, exact_acceleration).item(),
        exact_energy=0,
        mse=0,
        me=0,
        _count=num_graphs,
    )

    return main_loss, loss_raport


def loss_normalized_obstacle_scatter1(
    acceleration: torch.Tensor,
    # forces: torch.Tensor,
    # lhs: torch.Tensor,
    # rhs: torch.Tensor,
    energy_args: EnergyObstacleArguments,
    graph_sizes_base: List[int],
    exact_acceleration: Optional[torch.Tensor],
):
    num_graphs = len(graph_sizes_base)
    # acceleration_vector_all = get_acceleration_vector(acceleration, graph_sizes_base)
    # exact_acceleration_vector_all = get_acceleration_vector(exact_acceleration, graph_sizes_base)

    acceleration_split = acceleration.split(graph_sizes_base)
    exact_acceleration_split = exact_acceleration.split(graph_sizes_base)

    all_loss_raport = LossRaport()
    all_main_loss = 0.0
    for batch_graph_index in range(num_graphs):
        acceleration_vector = nph.stack_column(acceleration_split[batch_graph_index])
        exact_acceleration_vector = nph.stack_column(exact_acceleration_split[batch_graph_index])
        args = energy_args[0]
        # vmap
        inner_energy_torch = (
            energy_obstacle_torch(acceleration_vector=acceleration_vector, args=args) / num_graphs
        )
        inner_energy = (
            energy_obstacle_jax(
                acceleration_vector=jnp.asarray(acceleration_vector.cpu().detach()), args=args
            )
            / num_graphs
        )
        exact_inner_energy = (
            energy_obstacle_jax(
                acceleration_vector=jnp.asarray(exact_acceleration_vector.cpu().detach()), args=args
            )
            / num_graphs
        )
        boundary_integral = torch.tensor([0]) / num_graphs
        loss_energy = inner_energy  # + boundary_integral

        loss_mean = torch.tensor([0]) / num_graphs  # torch.mean(all_loss_mean)
        main_loss = thh.mean_error_torch(acceleration, exact_acceleration)  # loss_energy

        loss_raport = LossRaport(
            main=main_loss.item(),
            inner_energy=inner_energy.item(),
            energy=loss_energy.item(),
            boundary_integral=boundary_integral.item(),
            mean=loss_mean.item(),
            exact_energy=exact_inner_energy.item(),
            mse=thh.mean_square_error_torch(acceleration, exact_acceleration).item(),
            me=thh.mean_error_torch(acceleration, exact_acceleration).item(),
            _count=num_graphs,
        )
        loss_raport.relative_energy = loss_raport.energy - loss_raport.exact_energy

        all_loss_raport.add(loss_raport, normalize=False)
        all_main_loss += main_loss

    all_loss_raport.normalize()
    all_main_loss /= num_graphs
    return all_main_loss, all_loss_raport
