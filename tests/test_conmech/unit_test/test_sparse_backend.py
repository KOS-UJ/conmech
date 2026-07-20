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
# USA."""The assembled operators are sparse; check they are correct and Schur agrees

import numpy as np
import scipy.sparse

from conmech.mesh.mesh import Mesh
from conmech.simulations.problem_solver import Body
from conmech.dynamics.dynamics import Dynamics
from conmech.properties.body_properties import ViscoelasticProperties
from conmech.scene.body_forces import BodyForces
from conmech.dynamics.factory.dynamics_factory_method import get_factory
from conmech.mesh import mesh_builders
from conmech.properties.mesh_description import RectangleMeshDescription
from conmech.solvers.optimization.schur_complement import (
    calculate_schur_complement_matrices,
)


def _build_dynamics():
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_descr=RectangleMeshDescription(
            initial_position=None, max_element_perimeter=2 / 3, scale=[2, 3]
        )
    )
    mesh = object.__new__(Mesh)
    mesh.nodes = initial_nodes
    mesh.elements = elements

    body = object.__new__(Body)
    body.mesh = mesh
    body.properties = ViscoelasticProperties(mass_density=1.0, mu=4, lambda_=4, theta=4, zeta=4)

    dynamics = object.__new__(Dynamics)
    dynamics.body = body
    body.dynamics = dynamics
    dynamics.force = BodyForces(body)
    dynamics.temperature = BodyForces(body)
    dynamics.factory = get_factory(2)
    dynamics._Dynamics__relaxation = None
    dynamics._Dynamics__relaxation_tensor = None
    dynamics.reinitialize_matrices()
    return dynamics


def test_operators_are_sparse():
    dynamics = _build_dynamics()
    for name in ("elasticity", "viscosity", "acceleration_operator"):
        operator = getattr(dynamics, name)
        assert scipy.sparse.issparse(operator.data)
    # Sparsity actually saves memory relative to the dense N**2 storage.
    assert dynamics.elasticity.data.nnz < np.prod(dynamics.elasticity.data.shape)


def test_schur_complement_matches_dense_reference():
    dynamics = _build_dynamics()
    matrix = dynamics.elasticity.data  # sparse (2N, 2N)
    nodes_count = matrix.shape[0] // 2
    contact = slice(0, 5)
    free = slice(5, nodes_count)

    boundary, free_x_contact, contact_x_free, free_x_free_inv = calculate_schur_complement_matrices(
        matrix, 2, contact, free
    )

    dense = matrix.toarray()

    def dofs(node_slice):
        nodes = np.arange(nodes_count)[node_slice]
        return np.concatenate([k * nodes_count + nodes for k in range(2)])

    fd, cd = dofs(free), dofs(contact)
    ff = dense[np.ix_(fd, fd)]
    fc = dense[np.ix_(fd, cd)]
    cf = dense[np.ix_(cd, fd)]
    cc = dense[np.ix_(cd, cd)]
    ref_boundary = cc - cf @ np.linalg.solve(ff, fc)

    np.testing.assert_allclose(np.asarray(boundary), ref_boundary, atol=1e-8)
    np.testing.assert_allclose(free_x_contact.toarray(), fc, atol=1e-10)
    np.testing.assert_allclose(contact_x_free.toarray(), cf, atol=1e-10)

    rhs = np.arange(fc.shape[0], dtype=float).reshape(-1, 1)
    np.testing.assert_allclose(free_x_free_inv @ rhs, np.linalg.solve(ff, rhs), atol=1e-8)
