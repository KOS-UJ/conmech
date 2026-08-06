# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.

import numpy as np
import pytest

from dataclasses import dataclass
from typing import Optional, Type

import examples.Makela_et_al_1998 as makela

from conmech.dynamics.contact.contact_law import PotentialOfContactLaw
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.mesh_description import RectangleMeshDescription
from conmech.scenarios.problems import PoissonProblem
from conmech.simulations.problem_solver import PoissonSolver
from conmech.solvers.solver_methods import NO_VOLUME_MULTIPLIER
from conmech.dynamics.contact.contact_law import ContactLaw
from conmech.simulations.problem_solver import StaticSolver

B_COEF = 5.0


def make_law(kind):
    class Law(PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(var_nu, static_displacement_nu, dt):
            r = var_nu
            if kind == "quadratic":
                return 0.5 * (r - B_COEF) ** 2
            return (r - B_COEF) ** 2 if r < B_COEF else 2 * np.log(r - B_COEF + 1)

        @staticmethod
        def subderivative_normal_direction(var_nu, static_displacement_nu, dt):
            r = var_nu
            if kind == "quadratic":
                return r - B_COEF
            return 2 * (r - B_COEF) if r < B_COEF else 2 / (r - B_COEF + 1)

    return Law


def scalar_solver(kind):
    @dataclass
    class Setup(PoissonProblem):
        contact_law_2: Optional[Type[ContactLaw]] = None

    boundaries = BoundariesDescription(
        dirichlet=(lambda x: np.isclose(x[1], 0.0), lambda x: np.zeros(x.shape[0])),
        contact=lambda x: np.isclose(x[1], 1.0),
    )
    setup = Setup(
        mesh_descr=RectangleMeshDescription(
            initial_position=None, max_element_perimeter=0.5, scale=[2, 1]
        ),
        boundaries=boundaries,
    )
    setup.contact_law_2 = make_law(kind)
    setup.internal_temperature = staticmethod(lambda x, t=None: np.array([-4.0]))
    setup.outer_temperature = staticmethod(lambda x, t=None: np.array([0.0]))
    return PoissonSolver(setup, "schur").second_step_solver


def relative_error(solver, variable, sample):
    solver.iterate()
    length = len(variable)
    args = (
        variable,
        solver.body.mesh.nodes,
        solver.body.mesh.contact_boundary,
        solver.body.mesh.boundaries.contact_normals,
        solver.lhs,
        solver.rhs if len(np.shape(solver.rhs)) == 1 else solver.rhs[0],
        np.squeeze(np.zeros(length).reshape(1, -1)),
        NO_VOLUME_MULTIPLIER,
        solver.time_step,
    )
    analytic = np.asarray(solver.subgradient(sample, *args), dtype=float)

    def loss(vector):
        return float(np.ravel(solver.loss(vector, *args))[0])

    step = 1e-6
    numeric = np.empty_like(sample)
    for index in range(len(sample)):
        plus, minus = sample.copy(), sample.copy()
        plus[index] += step
        minus[index] -= step
        numeric[index] = (loss(plus) - loss(minus)) / (2 * step)
    return np.abs(analytic - numeric).max() / max(np.abs(numeric).max(), 1e-30)


# `1/(r-b+1)` is smooth below `b` and the quadratic law is smooth everywhere, so
# both samples stay on one branch and the central difference is meaningful
@pytest.mark.parametrize("kind, low, high", [("quadratic", 3.0, 6.0), ("logarithmic", 3.0, 4.8)])
def test_the_scalar_subgradient_is_the_gradient_of_its_loss(kind, low, high):
    """
    A scalar unknown has no direction to project onto.
    """
    solver = scalar_solver(kind)
    sample = np.linspace(low, high, solver.lhs.shape[0])
    assert relative_error(solver, solver.t_vector, sample) < 1e-7


def test_the_scalar_contact_term_is_present_at_all():
    solver = scalar_solver("quadratic")
    solver.iterate()
    length = len(solver.t_vector)
    args = (
        solver.t_vector,
        solver.body.mesh.nodes,
        solver.body.mesh.contact_boundary,
        solver.body.mesh.boundaries.contact_normals,
        solver.lhs,
        solver.rhs if len(np.shape(solver.rhs)) == 1 else solver.rhs[0],
        np.squeeze(np.zeros(length).reshape(1, -1)),
        NO_VOLUME_MULTIPLIER,
        solver.time_step,
    )
    sample = np.linspace(3.0, 6.0, solver.lhs.shape[0])
    analytic = np.asarray(solver.subgradient(sample, *args), dtype=float)
    without_contact = np.asarray(solver.lhs, dtype=float) @ sample - np.asarray(
        args[5], dtype=float
    )
    contact = np.abs(analytic - without_contact).max()
    assert contact > 0.1 * np.abs(without_contact).max(), contact


def test_the_vector_subgradient_is_the_gradient_of_its_loss():
    setup = makela.StaticSetup(
        mesh_descr=RectangleMeshDescription(
            initial_position=None, max_element_perimeter=0.5, scale=[8, 4]
        )
    )
    solver = StaticSolver(setup, "schur").step_solver
    sample = 0.05 * np.random.default_rng(0).standard_normal(solver.lhs.shape[0])
    assert relative_error(solver, solver.u_vector, sample) < 1e-6
