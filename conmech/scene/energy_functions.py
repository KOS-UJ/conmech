from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from conmech.dynamics.dynamics import _get_deform_grad
from conmech.helpers import nph


def _get_penetration_positive(displacement_step, normals, penetration):
    projection = nph.elementwise_dot(displacement_step, normals, keepdims=True) + penetration
    return (projection > 0) * projection


def _obstacle_resistance_potential_normal(penetration_norm, hardness, time_step):
    return hardness * 0.5 * (penetration_norm**2) * ((1.0 / time_step) ** 2)


def _obstacle_resistance_potential_tangential(
    penetration_norm,
    tangential_velocity,
    friction,
    time_step,
):
    return (
        (penetration_norm > 0)
        * friction
        # * nph.euclidean_norm(tangential_velocity, keepdims=True) #############################
        * jnp.log(nph.euclidean_norm(tangential_velocity, keepdims=True) + 1.0)
        * (1.0 / time_step)
    )


class EnergyObstacleArguments(NamedTuple):
    lhs_acceleration_jax: np.ndarray
    rhs_acceleration: np.ndarray
    boundary_velocity_old: np.ndarray
    boundary_normals: np.ndarray
    boundary_obstacle_normals: np.ndarray
    penetration: np.ndarray
    surface_per_boundary_node: np.ndarray
    body_prop: np.ndarray
    obstacle_prop: np.ndarray
    time_step: float
    element_initial_volume: np.ndarray
    dx_big_jax: np.ndarray
    base_displacement: np.ndarray
    base_velocity: np.ndarray
    base_energy_displacement: np.ndarray
    base_energy_velocity: np.ndarray


def _get_boundary_integral(
    acceleration,
    args: EnergyObstacleArguments,
):
    boundary_nodes_count = args.boundary_velocity_old.shape[0]
    boundary_a = acceleration[:boundary_nodes_count, :]  # TODO: boundary slice

    boundary_v_new = args.boundary_velocity_old + args.time_step * boundary_a
    boundary_displacement_step = args.time_step * boundary_v_new

    normals = args.boundary_normals
    nodes_volume = args.surface_per_boundary_node
    hardness = args.obstacle_prop.hardness
    friction = args.obstacle_prop.friction

    penetration_norm = _get_penetration_positive(
        displacement_step=boundary_displacement_step,
        normals=normals,
        penetration=args.penetration,
    )
    velocity_tangential = nph.get_tangential(boundary_v_new, normals)

    resistance_normal = _obstacle_resistance_potential_normal(
        penetration_norm=penetration_norm, hardness=hardness, time_step=args.time_step
    )
    resistance_tangential = _obstacle_resistance_potential_tangential(
        penetration_norm=args.penetration,
        tangential_velocity=velocity_tangential,
        friction=friction,
        time_step=args.time_step,
    )
    boundary_integral = (nodes_volume * (resistance_normal + resistance_tangential)).sum()
    return boundary_integral


def _get_strain_lin(deform_grad):
    dimension = deform_grad.shape[1]
    identity = jnp.eye(dimension)
    deform_grad_t = deform_grad.transpose((0, 2, 1))
    return 0.5 * (deform_grad + deform_grad_t) - identity


def _get_strain_green(deform_grad):
    dimension = deform_grad.shape[1]
    identity = jnp.eye(dimension)
    deform_grad_t = deform_grad.transpose((0, 2, 1))
    return 0.5 * (deform_grad_t @ deform_grad - identity)


def _compute_component_energy(
    component,
    dx_big_jax,
    element_initial_volume,
    prop_1,
    prop_2,
    use_green_strain,
):
    f_w = _get_deform_grad(component, dx_big_jax)
    if use_green_strain:
        eps_w = _get_strain_green(deform_grad=f_w)
    else:
        eps_w = _get_strain_lin(deform_grad=f_w)

    phi = prop_1 * (eps_w * eps_w).sum(axis=(1, 2)) + (prop_2 / 2.0) * (
        ((eps_w).trace(axis1=1, axis2=2) ** 2)
    )
    energy = element_initial_volume @ phi
    return energy


def _compute_displacement_energy(
    displacement, dx_big_jax, element_initial_volume, body_prop, use_green_strain
):
    return _compute_component_energy(
        component=displacement,
        dx_big_jax=dx_big_jax,
        element_initial_volume=element_initial_volume,
        prop_1=body_prop.mu,
        prop_2=body_prop.lambda_,
        use_green_strain=use_green_strain,
    )


def _compute_velocity_energy(
    velocity, dx_big_jax, element_initial_volume, body_prop, use_green_strain
):
    return _compute_component_energy(
        component=velocity,
        dx_big_jax=dx_big_jax,
        element_initial_volume=element_initial_volume,
        prop_1=body_prop.theta,
        prop_2=body_prop.zeta,
        use_green_strain=use_green_strain,
    )


def _compute_energy(acceleration, args, use_green_strain):
    new_displacement = args.base_displacement + acceleration * args.time_step**2
    new_velocity = args.base_velocity + acceleration * args.time_step

    energy_new = (
        _compute_displacement_energy(
            displacement=new_displacement,
            dx_big_jax=args.dx_big_jax,
            element_initial_volume=args.element_initial_volume,
            body_prop=args.body_prop,
            use_green_strain=use_green_strain,
        )
        - args.base_energy_displacement
    ) / (args.time_step**2)

    energy_new += (
        _compute_velocity_energy(
            velocity=new_velocity,
            dx_big_jax=args.dx_big_jax,
            element_initial_volume=args.element_initial_volume,
            body_prop=args.body_prop,
            use_green_strain=use_green_strain,
        )
        - args.base_energy_velocity
    ) / (args.time_step)

    return energy_new


def _energy_vector(value_vector, lhs, rhs):
    first = 0.5 * (lhs @ value_vector) - rhs
    value = first.reshape(-1) @ value_vector
    return value[0]


def _energy_obstacle(acceleration_vector, args: EnergyObstacleArguments, use_green_strain: bool):
    # TODO: Repeat if collision
    dimension = args.base_displacement.shape[1]
    main_energy0 = _energy_vector(
        value_vector=nph.stack_column(acceleration_vector),
        lhs=args.lhs_acceleration_jax,
        rhs=args.rhs_acceleration,
    )
    main_energy1 = _compute_energy(
        acceleration=nph.unstack(acceleration_vector, dim=dimension),
        args=args,
        use_green_strain=use_green_strain,
    )
    return main_energy0 + main_energy1


def _energy_obstacle_colliding(
    acceleration_vector, args: EnergyObstacleArguments, use_green_strain: bool
):
    # TODO: Repeat if collision
    dimension = args.base_displacement.shape[1]
    main_energy = _energy_obstacle(
        acceleration_vector=acceleration_vector, args=args, use_green_strain=use_green_strain
    )
    acceleration = nph.unstack(acceleration_vector, dim=dimension)
    boundary_integral = _get_boundary_integral(acceleration=acceleration, args=args)
    return main_energy + boundary_integral


# @partial(jax.jit, static_argnums=(2,))
# energy_obstacle_jax = jax.jit(energy_obstacle)

# hes_energy_obstacle_new = jax.jit(
#     lambda x, args: (lambda f, x, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x, args), v))(x))(
#         energy_obstacle, x, x
#     )
# )

# hes_energy_obstacle_colliding_new = jax.jit(
#     lambda x, args: (lambda f, x, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x, args), v))(x))(
#         energy_obstacle_colliding_jax, x, x
#     )
# )


@dataclass
class EnergyFunctions:
    def __init__(self, use_green_strain):
        self.energy_obstacle_jax = jax.jit(
            lambda acceleration_vector, args: _energy_obstacle(
                acceleration_vector=acceleration_vector,
                args=args,
                use_green_strain=use_green_strain,
            )
        )
        self.energy_obstacle_colliding_jax = jax.jit(
            lambda acceleration_vector, args: _energy_obstacle_colliding(
                acceleration_vector=acceleration_vector,
                args=args,
                use_green_strain=use_green_strain,
            )
        )
        self.compute_displacement_energy_jax = jax.jit(
            lambda displacement, dx_big_jax, element_initial_volume, body_prop: _compute_displacement_energy(
                displacement=displacement,
                dx_big_jax=dx_big_jax,
                element_initial_volume=element_initial_volume,
                body_prop=body_prop,
                use_green_strain=use_green_strain,
            )
        )
        self.compute_velocity_energy_jax = jax.jit(
            lambda velocity, dx_big_jax, element_initial_volume, body_prop: _compute_velocity_energy(
                velocity=velocity,
                dx_big_jax=dx_big_jax,
                element_initial_volume=element_initial_volume,
                body_prop=body_prop,
                use_green_strain=use_green_strain,
            )
        )

    def get_energy_function(self, scene):
        if not scene.is_colliding():
            return self.energy_obstacle_jax
        else:
            return self.energy_obstacle_colliding_jax
