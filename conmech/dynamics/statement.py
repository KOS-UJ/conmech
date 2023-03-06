from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Variables:
    absement: Optional[np.ndarray] = None  # https://en.wikipedia.org/wiki/Absement
    displacement: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    electric_potential: Optional[np.ndarray] = None
    time_step: Optional[float] = None


class Statement:
    def __init__(self, body, dimension):
        self.body = body
        self.dimension = dimension
        self.left_hand_side = None
        self.right_hand_side = None
        self.dirichlet_cond_name = "dirichlet"

    def update_left_hand_side(self, var: Variables):
        raise NotImplementedError()

    def update_right_hand_side(self, var: Variables):
        raise NotImplementedError()

    def update(self, var: Variables):
        self.update_left_hand_side(var)
        self.update_right_hand_side(var)
        self.apply_dirichlet_condition()

    def apply_dirichlet_condition(self):
        for dirichlet_cond in self.find_dirichlet_conditions():
            c = self.body.mesh.boundaries.boundaries[dirichlet_cond].node_condition
            node_count = self.body.mesh.nodes_count
            for i, j in self.body.mesh.boundaries.get_all_boundary_indices(
                dirichlet_cond, node_count, self.dimension
            ):
                self.right_hand_side[:] -= self.left_hand_side[:, i] @ c[j]
                self.left_hand_side[:, i] = 0
                self.left_hand_side[i, :] = 0
                # have to be "[i][:, i]" instead of just a "[i, i]" because the i may be ndarray
                self.left_hand_side[i][:, i] = np.eye(j.stop - j.start)
                self.right_hand_side[i] = c[j]

    def find_dirichlet_conditions(self):
        boundaries = self.body.mesh.boundaries.boundaries
        if self.dirichlet_cond_name in boundaries:
            yield self.dirichlet_cond_name
            return
        i = 0
        while True:
            next_name = self.dirichlet_cond_name + "_" + str(i)
            if next_name in boundaries:
                yield next_name
            else:
                return
            i += 1


class StaticDisplacementStatement(Statement):
    def __init__(self, dynamics):
        super().__init__(dynamics, 2)

    def update_left_hand_side(self, var: Variables):
        self.left_hand_side = self.body.elasticity.copy()

    def update_right_hand_side(self, var: Variables):
        self.right_hand_side = self.body.get_integrated_forces_vector()


class QuasistaticLongMemoryStatement(Statement):
    def __init__(self, body):
        super().__init__(body, 2)

    def update_left_hand_side(self, var: Variables):
        assert var.time_step is not None

        self.left_hand_side = self.body.elasticity.copy() + self.body.long_memory * var.time_step

    def update_right_hand_side(self, var: Variables):
        assert var.absement is not None

        self.right_hand_side = (
                self.body.get_integrated_forces_vector() - self.body.long_memory @ var.absement.T
        )


class QuasistaticVelocityStatement(Statement):
    def __init__(self, body):
        super().__init__(body, 2)

    def update_left_hand_side(self, var: Variables):
        assert var.time_step is not None

        self.left_hand_side = self.body.viscosity.copy() + self.body.elasticity * var.time_step

    def update_right_hand_side(self, var: Variables):
        assert var.displacement is not None

        self.right_hand_side = (
            self.body.get_integrated_forces_vector() - self.body.elasticity @ var.displacement.T
        )


class DynamicVelocityStatement(Statement):
    def __init__(self, body):
        super().__init__(body, 2)

    def update_left_hand_side(self, var):
        assert var.time_step is not None

        self.left_hand_side = (
            self.body.viscosity
            + (1 / var.time_step) * self.body.acceleration_operator
            + self.body.elasticity * var.time_step
        )

    def update_right_hand_side(self, var):
        assert var.displacement is not None
        assert var.velocity is not None
        assert var.time_step is not None

        A = -1 * self.body.elasticity @ var.displacement

        A += (1 / var.time_step) * self.body.acceleration_operator @ var.velocity

        self.right_hand_side = self.body.get_integrated_forces_vector() + A


class DynamicVelocityWithTemperatureStatement(DynamicVelocityStatement):
    def update_right_hand_side(self, var):
        super().update_right_hand_side(var)

        assert var.temperature is not None

        A = self.body.thermal_expansion.T @ var.temperature

        self.right_hand_side += A


class TemperatureStatement(Statement):
    def __init__(self, body):
        super().__init__(body, 1)

    def update_left_hand_side(self, var):
        assert var.time_step is not None

        ind = self.body.mesh.nodes_count  # 1 dimensional

        self.left_hand_side = (1 / var.time_step) * self.body.acceleration_operator[
            :ind, :ind
        ] + self.body.thermal_conductivity[:ind, :ind]

    def update_right_hand_side(self, var):
        assert var.velocity is not None
        assert var.time_step is not None
        assert var.temperature is not None

        rhs = (-1) * self.body.thermal_expansion @ var.velocity

        ind = self.body.mesh.nodes_count  # 1 dimensional

        rhs += (1 / var.time_step) * self.body.acceleration_operator[:ind, :ind] @ var.temperature
        self.right_hand_side = rhs


class PiezoelectricStatement(Statement):
    def __init__(self, body):
        super().__init__(body, 1)
        self.dirichlet_cond_name = "piezo_" + self.dirichlet_cond_name

    def update_left_hand_side(self, var):
        self.left_hand_side = self.body.permittivity.copy()

    def update_right_hand_side(self, var):
        assert var.displacement is not None

        rhs = self.body.piezoelectricity @ var.displacement
        self.right_hand_side = rhs


class QuasistaticVelocityWithPiezoelectricStatement(QuasistaticVelocityStatement):
    def update_right_hand_side(self, var):
        super().update_right_hand_side(var)

        assert var.electric_potential is not None

        A = (-1) * self.body.piezoelectricity.T @ var.electric_potential

        self.right_hand_side += A


class DynamicVelocityWithPiezoelectricStatement(DynamicVelocityStatement):
    def update_right_hand_side(self, var):
        super().update_right_hand_side(var)

        assert var.electric_potential is not None

        A = self.body.piezoelectricity.T @ var.electric_potential

        self.right_hand_side += A
