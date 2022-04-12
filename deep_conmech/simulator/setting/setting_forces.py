from conmech.dynamics.dynamics import Dynamics
from conmech.helpers import nph
from conmech.solvers.optimization.schur_complement import SchurComplement


def energy_new(a, C, E):
    a_vector = nph.stack_column(a)
    first = 0.5 * (C @ a_vector) - E
    value = first.reshape(-1) @ a_vector
    return value


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

    def get_all_normalized_E_np(self, t):
        normalized_E = self.get_normalized_E_np(t)
        (
            normalized_E_boundary,
            normalized_E_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=normalized_E,
            dimension=self.dimension,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.solver_cache.free_x_free_inverted,
            contact_x_free=self.solver_cache.contact_x_free,
        )
        return normalized_E_boundary, normalized_E_free

    def get_normalized_E_np(self, t):
        return self.get_E(
            forces=self.normalized_forces,
            displacement_old=self.normalized_displacement_old,
            velocity_old=self.normalized_velocity_old,
            const_volume=self.volume,
            elasticity=self.elasticity,
            viscosity=self.viscosity,
            time_step=self.time_step,
        )

    def get_E(
        self,
        forces,
        displacement_old,
        velocity_old,
        const_volume,
        elasticity,
        viscosity,
        time_step,
    ):
        displacement_old_vector = nph.stack_column(displacement_old)
        velocity_old_vector = nph.stack_column(velocity_old)

        F_vector = nph.stack_column(const_volume @ forces)
        E = (
            F_vector
            - (viscosity + elasticity * time_step) @ velocity_old_vector
            - elasticity @ displacement_old_vector
        )
        return E

    @property
    def input_forces(self):
        return self.normalized_forces
