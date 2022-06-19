from typing import List, Optional

import numpy as np
import torch

from conmech.helpers import nph
from conmech.scene.body_forces import energy, energy_vector
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
    forces: torch.Tensor,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    energy_args: EnergyObstacleArgumentsTorch,
    graph_sizes_base: List[int],
    exact_acceleration: Optional[torch.Tensor],
):
    num_graphs = len(graph_sizes_base)
    acceleration_vector = get_acceleration_vector(acceleration, graph_sizes_base)
    exact_acceleration_vector = get_acceleration_vector(exact_acceleration, graph_sizes_base)

    inner_energy = energy_vector(value_vector=acceleration_vector, lhs=lhs, rhs=rhs) / num_graphs
    exact_inner_energy = (
        energy_vector(value_vector=exact_acceleration_vector, lhs=lhs, rhs=rhs) / num_graphs
    )
    boundary_integral = torch.tensor([0]) / num_graphs
    loss_energy = inner_energy  # + boundary_integral

    loss_mean = torch.tensor([0]) / num_graphs  # torch.mean(all_loss_mean)
    main_loss = loss_energy

    loss_raport = LossRaport(
        main=main_loss.item(),
        inner_energy=inner_energy.item(),
        energy=loss_energy.item(),
        boundary_integral=boundary_integral.item(),
        mean=loss_mean.item(),
        exact_energy=exact_inner_energy.item(),
        mse=thh.mse_torch(acceleration, exact_acceleration).item(),
        me=thh.me_torch(acceleration, exact_acceleration).item(),
        _count=num_graphs,
    )
    loss_raport.relative_energy = loss_raport.energy - loss_raport.exact_energy

    return main_loss, loss_raport


def loss_normalized_obstacle(
    acceleration: torch.Tensor,
    forces: torch.Tensor,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    energy_args: EnergyObstacleArgumentsTorch,
    exact_acceleration: Optional[torch.Tensor],
):
    inner_energy = energy(value=acceleration, lhs=lhs, rhs=rhs)
    exact_inner_energy = energy(value=exact_acceleration, lhs=lhs, rhs=rhs)

    # boundary_integral = get_boundary_integral(acceleration=acceleration, args=energy_args)
    boundary_integral = torch.tensor([0])
    loss_energy = inner_energy  # + boundary_integral

    main_loss = loss_energy  # mse  # loss_mean + 0.01 * loss_energy # loss_energy

    loss_raport = LossRaport(
        main=main_loss.item(),
        inner_energy=inner_energy.item(),
        energy=loss_energy.item(),
        boundary_integral=boundary_integral.item(),
        mean=torch.tensor([0]).item(),
        exact_energy=exact_inner_energy.item(),
        mse=thh.mse_torch(acceleration, exact_acceleration).item(),
        me=thh.me_torch(acceleration, exact_acceleration).item(),
        _count=1,
    )

    loss_raport.relative_energy = loss_raport.energy - loss_raport.exact_energy
    loss_raport.normalized_relative_energy = loss_raport.relative_energy / np.abs(
        loss_raport.exact_energy
    )

    return main_loss, loss_raport
