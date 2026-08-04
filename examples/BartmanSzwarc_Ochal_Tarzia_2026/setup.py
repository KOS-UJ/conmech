# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2025-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np

from conmech.dynamics.contact.contact_law import ContactLaw, PotentialOfContactLaw
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.mesh_description import NestedRectangleMeshDescription
from conmech.scenarios.problems import PoissonProblem

OUTPUTS_PATH = "./output/BOT2026"

B_COEF = 5.0
GAMMA1_VALUE = 0.0
GEOM_TOL = 1e-12

#: Meshes compared against a discrete reference. Nested, so a P1 function of one
#: is represented exactly on the next and the comparison carries no
#: interpolation error of its own.
IHS = [4, 8, 16, 32]
#: Reference mesh for the examples without a closed-form solution.
IH_REF = 128
#: Meshes for the examples where the reference is the formula itself.
IHS_EXACT = [4, 8, 16, 32, 64]

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10_000, 100_000, 1_000_000, np.inf]
#: Finite values for the `O(1/alpha)` study.
ALPHAS_FINITE = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000]
#: Columns of the `h` by `alpha` matrix.
ALPHAS_MATRIX = [1, 10, 100, 1000, 10_000]
#: Values whose `Gamma_3` trace is drawn.
ALPHAS_TRACE = [1, 10, 100, 1000, 10_000, 100_000]

#: Amplitude of the oscillatory part of the 2D closed-form solution, chosen so
#: that it contributes on the same order as the polynomial part.
A_COEF = 1.0

TEMPERATURE_GRID = (
    ((np.inf, 4), (np.inf, 32)),
    ((10, 32), (100, 32)),
    ((0.1, 32), (1, 32)),
    ((0.01, 4), (0.01, 32)),
)


def mesh_description(ih: int) -> NestedRectangleMeshDescription:
    return NestedRectangleMeshDescription(initial_position=None, cells_per_unit=ih, scale=[2, 1])


def mesh_size(ih: int) -> float:
    return 1.0 / ih


def alpha_tag(alpha: float) -> str:
    return "inf" if np.isinf(alpha) else f"{alpha:g}"


def on_gamma1(x: np.ndarray) -> bool:
    return bool(np.isclose(x[1], 0.0, atol=GEOM_TOL))


def on_gamma3(x: np.ndarray) -> bool:
    return bool(np.isclose(x[1], 1.0, atol=GEOM_TOL))


def make_quadratic_contact_law(slope: float, b: float = B_COEF) -> Type[ContactLaw]:

    class QuadraticContactLaw(PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            return slope * 0.5 * (var_nu - b) ** 2

        @staticmethod
        def subderivative_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            return slope * (var_nu - b)

    return QuadraticContactLaw


def make_slope_contact_law(slope: float, b: float = B_COEF) -> Type[ContactLaw]:
    """
    Non-convex superpotential
    """

    class TarziaContactLaw(PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            r = var_nu
            # EXAMPLE 11
            if r < b:
                result = (r - b) ** 2
            else:
                result = 2 * np.log((r - b) + 1)
            return slope * result

        @staticmethod
        def subderivative_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            r = var_nu
            # EXAMPLE 11
            if r < b:
                result = 2 * (r - b)
            else:
                result = 2 / (r - b + 1)
            return slope * result

    return TarziaContactLaw


def _g_constant(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
    return np.array([-4.0])


def _q_zero(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
    return np.array([0.0])


def _q_example2(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
    """`q = 32 y (y - 1)`"""
    y = x[1]
    return np.array([32.0 * y * (y - 1.0)])


def _g_example1_2d(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
    """`g = -4 + A (5 pi^2 / 4) sin(pi x / 2) sin(pi y)`"""
    return np.array(
        [-4.0 + A_COEF * (5.0 * np.pi**2 / 4.0) * np.sin(np.pi * x[0] / 2.0) * np.sin(np.pi * x[1])]
    )


def _q_example1_2d(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
    return np.array([-A_COEF * (np.pi / 2.0) * np.sin(np.pi * x[1])])


def alpha_slope(alpha: float) -> float:
    """`(3 alpha - 4) / (1 + alpha)`, the linear coefficient of `u_alpha`."""
    if not np.isfinite(alpha):
        return 3.0
    return (3.0 * alpha - 4.0) / (1.0 + alpha)


def u_exact_1d(alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    """`u_alpha(y) = 2 y^2 + ((3 alpha - 4)/(1 + alpha)) y`; `alpha = inf` allowed."""
    coefficient = alpha_slope(alpha)

    def u_exact(points: np.ndarray) -> np.ndarray:
        y = np.asarray(points, dtype=float)[:, 1]
        return 2.0 * y**2 + coefficient * y

    return u_exact


def grad_u_exact_1d(alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    coefficient = alpha_slope(alpha)

    def grad_u_exact(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        out = np.zeros((points.shape[0], 2))
        out[:, 1] = 4.0 * points[:, 1] + coefficient
        return out

    return grad_u_exact


def u_exact_2d(points: np.ndarray) -> np.ndarray:
    """`u = 2 y^2 + 3 y + A sin(pi x / 2) sin(pi y)`."""
    points = np.asarray(points, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 2.0 * y**2 + 3.0 * y + A_COEF * np.sin(np.pi * x / 2.0) * np.sin(np.pi * y)


def grad_u_exact_2d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    x, y = points[:, 0], points[:, 1]
    out = np.empty((points.shape[0], 2))
    out[:, 0] = A_COEF * (np.pi / 2.0) * np.cos(np.pi * x / 2.0) * np.sin(np.pi * y)
    out[:, 1] = 4.0 * y + 3.0 + A_COEF * np.pi * np.sin(np.pi * x / 2.0) * np.cos(np.pi * y)
    return out


def analytic_gap_l2(alpha: float) -> float:
    """`||u_alpha - u_inf||_L2 = (7/(1+alpha)) sqrt(2/3)`."""
    return (7.0 / (1.0 + alpha)) * np.sqrt(2.0 / 3.0)


def analytic_gap_h1(alpha: float) -> float:
    """`||grad(u_alpha - u_inf)||_L2 = (7/(1+alpha)) sqrt(2)`."""
    return (7.0 / (1.0 + alpha)) * np.sqrt(2.0)


def analytic_gap_v(alpha: float) -> float:
    return np.sqrt(analytic_gap_l2(alpha) ** 2 + analytic_gap_h1(alpha) ** 2)


def analytic_trace_1d(alpha: float) -> float:
    """`u_alpha(y = 1)`, which tends to `b`."""
    return 2.0 + alpha_slope(alpha)


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    b: float
    internal_temperature: Callable
    outer_temperature: Callable
    make_contact_law: Callable[[float], Type[ContactLaw]]
    gamma1_value: float = GAMMA1_VALUE
    is_gamma1: Callable[[np.ndarray], bool] = on_gamma1
    is_gamma3: Callable[[np.ndarray], bool] = on_gamma3
    exact_inf: Optional[Callable] = None
    grad_exact_inf: Optional[Callable] = None
    exact_alpha: Optional[Callable[[float], Callable]] = None
    grad_exact_alpha: Optional[Callable[[float], Callable]] = None

    def exact_for(self, alpha: float):
        if not np.isfinite(alpha):
            return self.exact_inf, self.grad_exact_inf
        if self.exact_alpha is None:
            return None, None
        return self.exact_alpha(alpha), self.grad_exact_alpha(alpha)


EXAMPLE_1_1D = ExampleSpec(
    name="example1_1d",
    b=B_COEF,
    internal_temperature=_g_constant,
    outer_temperature=_q_zero,
    make_contact_law=lambda alpha: make_quadratic_contact_law(alpha, B_COEF),
    exact_inf=u_exact_1d(np.inf),
    grad_exact_inf=grad_u_exact_1d(np.inf),
    exact_alpha=u_exact_1d,
    grad_exact_alpha=grad_u_exact_1d,
)

EXAMPLE_1_2D = ExampleSpec(
    name="example1_2d",
    b=B_COEF,
    internal_temperature=_g_example1_2d,
    outer_temperature=_q_example1_2d,
    make_contact_law=lambda alpha: make_quadratic_contact_law(alpha, B_COEF),
    exact_inf=u_exact_2d,
    grad_exact_inf=grad_u_exact_2d,
)

EXAMPLE_2 = ExampleSpec(
    name="example2",
    b=B_COEF,
    internal_temperature=_g_constant,
    outer_temperature=_q_example2,
    make_contact_law=lambda alpha: make_slope_contact_law(alpha, B_COEF),
)

EXAMPLES: Dict[str, ExampleSpec] = {
    spec.name: spec for spec in (EXAMPLE_1_1D, EXAMPLE_1_2D, EXAMPLE_2)
}


@dataclass
class StaticPoissonSetup(PoissonProblem):
    contact_law_2: Optional[Type[ContactLaw]] = None


def build_setup(spec: ExampleSpec, alpha: float, mesh_descr) -> Tuple[PoissonProblem, str]:
    is_gamma1 = spec.is_gamma1
    is_gamma3 = spec.is_gamma3

    if np.isinf(alpha):

        def is_dirichlet(x: np.ndarray) -> bool:
            return is_gamma1(x) or is_gamma3(x)

        def dirichlet_value(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if x.ndim != 2:
                raise ValueError(
                    f"expected the (n_nodes, 2) block of Dirichlet nodes, got shape {x.shape}"
                )
            on_top = np.array([is_gamma3(point) for point in x], dtype=bool)
            return np.where(on_top, spec.b, spec.gamma1_value)

        boundaries = BoundariesDescription(dirichlet=(is_dirichlet, dirichlet_value))
        contact_law = None
        solving_method = "direct"
    else:

        def gamma1_only(x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], spec.gamma1_value)

        boundaries = BoundariesDescription(dirichlet=(is_gamma1, gamma1_only), contact=is_gamma3)
        contact_law = spec.make_contact_law(alpha)
        solving_method = "schur"

    setup = StaticPoissonSetup(mesh_descr=mesh_descr, boundaries=boundaries)
    setup.contact_law_2 = contact_law
    setup.internal_temperature = spec.internal_temperature
    setup.outer_temperature = spec.outer_temperature
    return setup, solving_method
