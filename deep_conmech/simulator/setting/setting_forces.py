from conmech.helpers import nph
from conmech.solvers.optimization.schur_complement import SchurComplement
from deep_conmech.simulator.dynamics.dynamics import Dynamics


def L2_new(a, C, E):
    a_vector = nph.stack_column(a)
    first = 0.5 * (C @ a_vector) - E
    value = first.reshape(-1) @ a_vector
    return value


class SettingForces(Dynamics):
    def __init__(
            self, mesh_data, body_prop, schedule, normalize_by_rotation: bool, create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
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
        super().prepare()
        self.forces = forces

    def clear(self):
        super().clear()
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
            free_x_free_inverted=self.free_x_free_inverted,
            contact_x_free=self.contact_x_free,
        )
        return normalized_E_boundary, normalized_E_free

    def get_normalized_E_np(self, t):
        return self.get_E(
            forces=self.normalized_forces,
            u_old=self.normalized_u_old,
            v_old=self.normalized_v_old,
            const_volume=self.const_volume,
            const_elasticity=self.const_elasticity,
            const_viscosity=self.const_viscosity,
            time_step=self.time_step,
        )

    def get_E(
            self,
            forces,
            u_old,
            v_old,
            const_volume,
            const_elasticity,
            const_viscosity,
            time_step,
    ):
        u_old_vector = nph.stack_column(u_old)
        v_old_vector = nph.stack_column(v_old)

        F_vector = nph.stack_column(const_volume @ forces)
        E = (
                F_vector
                - (const_viscosity + const_elasticity * time_step) @ v_old_vector
                - const_elasticity @ u_old_vector
        )
        return E
