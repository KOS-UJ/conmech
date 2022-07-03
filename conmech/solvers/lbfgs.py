"""The Limited-Memory Broyden-Fletcher-Goldfarb-Shanno minimization algorithm."""
from functools import partial
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
from jax import lax
from jax._src.scipy.optimize.line_search import line_search

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)


Array = Any

##############################


class LBFGSResults(NamedTuple):
    """Results from L-BFGS optimization

    Parameters:
      converged: True if minimization converged
      failed: True if non-zero status and not converged
      k: integer number of iterations of the main loop (optimisation steps)
      nfev: integer total number of objective evaluations performed.
      ngev: integer total number of jacobian evaluations
      x_k: array containing the last argument value found during the search. If
        the search converged, then this value is the argmin of the objective
        function.
      f_k: array containing the value of the objective function at `x_k`. If the
        search converged, then this is the (local) minimum of the objective
        function.
      g_k: array containing the gradient of the objective function at `x_k`. If
        the search converged the l2-norm of this tensor should be below the
        tolerance.
      status: integer describing the status:
        0 = nominal  ,  1 = max iters reached  ,  2 = max fun evals reached
        3 = max grad evals reached  ,  4 = insufficient progress (ftol)
        5 = line search failed
      ls_status: integer describing the end status of the last line search
    """

    converged: Union[bool, Array]
    overrun: Union[bool, Array]
    xdiff_max: float
    xdiff_mean: float
    iter_count: Union[int, Array]
    nfev: Union[int, Array]
    ngev: Union[int, Array]
    x_k: Array
    f_k: Array
    g_k: Array
    s_history: Array
    y_history: Array
    rho_history: Array
    gamma: Union[float, Array]
    ls_status: Union[int, Array]


##############################


def minimize_lbfgs(
    fun: Callable,
    x0: Array,
    xtol_max: float,
    xtol_mean: float,
    max_iter: Optional[int],
    init_hes: Optional[Array] = None,
    maxcor: int = 10,
    maxls: int = 20,
):
    """
    Minimize a function using L-BFGS

    Implements the L-BFGS algorithm from
      Algorithm 7.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 176-185
    And generalizes to complex variables from
       Sorber, L., Barel, M.V. and Lathauwer, L.D., 2012.
       "Unconstrained optimization of real functions in complex variables"
       SIAM Journal on Optimization, 22(3), pp.879-898.

    Args:
      fun: function of the form f(x) where x is a flat ndarray and returns a real scalar.
        The function should be composed of operations with vjp defined.
      x0: initial guess
      maxiter: maximum number of iterations
      norm: order of norm for convergence check. Default inf.
      maxcor: maximum number of metric corrections ("history size")
      ftol: terminates the minimization when `(f_k - f_{k+1}) < ftol`
      gtol: terminates the minimization when `|g_k|_norm < gtol`
      maxfun: maximum number of function evaluations
      maxgrad: maximum number of gradient evaluations
      maxls: maximum number of line search steps (per iteration)

    Returns:
      Optimization results.
    """
    d = len(x0)
    dtype = jnp.dtype(x0)

    # initial evaluation
    f_0, g_0 = jax.value_and_grad(fun)(x0)
    state_initial = LBFGSResults(
        converged=False,
        overrun=False,
        xdiff_max=jnp.inf,
        xdiff_mean=jnp.inf,
        iter_count=0,
        nfev=1,
        ngev=1,
        x_k=x0,
        f_k=f_0,
        g_k=g_0,
        s_history=jnp.zeros((maxcor, d), dtype=dtype),
        y_history=jnp.zeros((maxcor, d), dtype=dtype),
        rho_history=jnp.zeros((maxcor,), dtype=dtype),
        gamma=1.0,
        ls_status=0,
    )

    def cond_fun(state: LBFGSResults):
        return (~state.converged) & (~state.overrun)

    def body_fun(state: LBFGSResults):
        # find search direction
        p_k = _two_loop_recursion(state, init_hes)

        # line search
        ls_results = line_search(
            f=fun,
            xk=state.x_k,
            pk=p_k,
            old_fval=state.f_k,
            gfk=state.g_k,
            maxiter=maxls,
        )

        # evaluate at next iterate
        s_k = ls_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = ls_results.f_k
        g_kp1 = ls_results.g_k
        y_k = g_kp1 - state.g_k
        rho_k_inv = jnp.real(_dot(y_k, s_k))
        rho_k = jnp.reciprocal(rho_k_inv)
        gamma = rho_k_inv / jnp.real(_dot(jnp.conj(y_k), y_k))

        # Added custom stop criterion
        normalized_step = jnp.abs(s_k)
        xdiff_max = jnp.max(normalized_step)
        xdiff_mean = jnp.mean(normalized_step)
        converged = (xdiff_max < xtol_max) & (xdiff_mean < xtol_mean)
        overrun = max_iter is not None and state.iter_count >= max_iter

        # TODO(jakevdp): use a fixed-point procedure rather than type-casting?
        state = state._replace(
            converged=converged,
            overrun=overrun,
            xdiff_max=xdiff_max,
            xdiff_mean=xdiff_mean,
            iter_count=state.iter_count + 1,
            nfev=state.nfev + ls_results.nfev,
            ngev=state.ngev + ls_results.ngev,
            x_k=x_kp1.astype(state.x_k.dtype),
            f_k=f_kp1.astype(state.f_k.dtype),
            g_k=g_kp1.astype(state.g_k.dtype),
            s_history=_update_history_vectors(history=state.s_history, new=s_k),
            y_history=_update_history_vectors(history=state.y_history, new=y_k),
            rho_history=_update_history_scalars(history=state.rho_history, new=rho_k),
            gamma=gamma,
            ls_status=ls_results.status,
        )

        return state

    state = lax.while_loop(cond_fun, body_fun, state_initial)
    return state


def _two_loop_recursion(state: LBFGSResults, hes: Optional[Array] = None):
    his_size = len(state.rho_history)
    curr_size = jnp.where(state.iter_count < his_size, state.iter_count, his_size)
    q = -jnp.conj(state.g_k)
    a_his = jnp.zeros_like(state.rho_history)

    def body_fun1(j, carry):
        i = his_size - 1 - j
        _q, _a_his = carry
        a_i = state.rho_history[i] * jnp.real(_dot(jnp.conj(state.s_history[i]), _q))
        _a_his = _a_his.at[i].set(a_i)
        _q = _q - a_i * jnp.conj(state.y_history[i])
        return _q, _a_his

    q, a_his = lax.fori_loop(0, curr_size, body_fun1, (q, a_his))

    # Added Hessian preconditioning
    if hes is None:
        q = state.gamma * q
    else:
        q, _ = jax.scipy.sparse.linalg.cg(A=hes, b=q)  # , M=M)

    def body_fun2(j, _q):
        i = his_size - curr_size + j
        b_i = state.rho_history[i] * jnp.real(_dot(state.y_history[i], _q))
        _q = _q + (a_his[i] - b_i) * state.s_history[i]
        return _q

    q = lax.fori_loop(0, curr_size, body_fun2, q)
    return q


def _update_history_vectors(history, new):
    # TODO(Jakob-Unfried) use rolling buffer instead? See #6053
    return jnp.roll(history, -1, axis=0).at[-1, :].set(new)


def _update_history_scalars(history, new):
    # TODO(Jakob-Unfried) use rolling buffer instead? See #6053
    return jnp.roll(history, -1, axis=0).at[-1].set(new)
