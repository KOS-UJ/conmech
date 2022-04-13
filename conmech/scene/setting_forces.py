from dataclasses import dataclass

import numpy as np

from conmech.dynamics.dynamics import Dynamics
from conmech.helpers import nph
from conmech.solvers.optimization.schur_complement import SchurComplement


def energy(value, lhs, rhs):
    value_vector = nph.stack_column(value)
    first = 0.5 * (lhs @ value_vector) - rhs
    value = first.reshape(-1) @ value_vector
    return value


@dataclass
class GetRhsArgs:
    forces: np.ndarray
    displacement_old: np.ndarray
    velocity_old: np.ndarray
    volume: np.ndarray
    elasticity: np.ndarray
    viscosity: np.ndarray
    time_step: float


def get_rhs(args: GetRhsArgs):
    displacement_old_vector = nph.stack_column(args.displacement_old)
    velocity_old_vector = nph.stack_column(args.velocity_old)
    f_vector = nph.stack_column(args.volume @ args.forces)
    rhs = (
        f_vector
        - (args.viscosity + args.elasticity * args.time_step) @ velocity_old_vector
        - args.elasticity @ displacement_old_vector
    )
    return rhs


class SettingForces(Dynamics):
    def __init__(
        self,
        mesh_prop,
        body_prop,
        schedule,
        normalize_by_rotation: bool,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        self.forces = None

    @property
    def normalized_forces(self):
        return self.normalize_rotate(self.forces)

    def prepare(self, forces):
        self.forces = forces

    def clear(self):
        self.forces = None

    def get_all_normalized_rhs_np(self):
        normalized_rhs = self.get_normalized_rhs_np()
        (
            normalized_rhs_boundary,
            normalized_rhs_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=normalized_rhs,
            dimension=self.dimension,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.solver_cache.free_x_free_inverted,
            contact_x_free=self.solver_cache.contact_x_free,
        )
        return normalized_rhs_boundary, normalized_rhs_free

    def get_normalized_rhs_np(self, temperature=None):
        _ = temperature
        args = GetRhsArgs(
            forces=self.normalized_forces,
            displacement_old=self.normalized_displacement_old,
            velocity_old=self.normalized_velocity_old,
            volume=self.volume,
            elasticity=self.elasticity,
            viscosity=self.viscosity,
            time_step=self.time_step,
        )
        return get_rhs(args)

    @property
    def input_forces(self):
        return self.normalized_forces
