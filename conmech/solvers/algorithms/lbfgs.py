"""The Limited-Memory Broyden-Fletcher-Goldfarb-Shanno minimization algorithm."""
# pylint: skip-file
import os
from functools import partial
from time import time
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
from jax import lax
from jax._src.scipy.optimize.line_search import line_search

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)

bits = 64


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    C = fpa
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    d1 = jnp.array([[dc**2, -(db**2)], [-(dc**3), db**3]])
    A, B = _dot(d1, jnp.array([fb - fa - C * db, fc - fa - C * dc])) / denom

    radical = B * B - 3.0 * A * C
    xmin = a + (-B + jnp.sqrt(radical)) / (3.0 * A)

    return xmin


def _quadmin(a, fa, fpa, b, fb):
    D = fa
    C = fpa
    db = b - a
    B = (fb - D - C * db) / (db**2)
    xmin = a - C / (2.0 * B)
    return xmin


def _binary_replace(replace_bit, original_dict, new_dict, keys=None):
    if keys is None:
        keys = new_dict.keys()
    out = dict()
    for key in keys:
        out[key] = jnp.where(replace_bit, new_dict[key], original_dict[key])
    return out


class _ZoomState(NamedTuple):
    done: Union[bool, jnp.ndarray]
    failed: Union[bool, jnp.ndarray]
    j: Union[int, jnp.ndarray]
    a_lo: Union[float, jnp.ndarray]
    phi_lo: Union[float, jnp.ndarray]
    dphi_lo: Union[float, jnp.ndarray]
    a_hi: Union[float, jnp.ndarray]
    phi_hi: Union[float, jnp.ndarray]
    dphi_hi: Union[float, jnp.ndarray]
    a_rec: Union[float, jnp.ndarray]
    phi_rec: Union[float, jnp.ndarray]
    a_star: Union[float, jnp.ndarray]
    phi_star: Union[float, jnp.ndarray]
    dphi_star: Union[float, jnp.ndarray]
    g_star: Union[float, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]


def _zoom(
    restricted_func_and_grad,
    wolfe_one_neg,
    wolfe_two,
    a_lo,
    phi_lo,
    dphi_lo,
    a_hi,
    phi_hi,
    dphi_hi,
    g_0,
    pass_through,
    maxiter_zoom,
):
    """
    Implementation of zoom. Algorithm 3.6 from Wright and Nocedal, 'Numerical
    Optimization', 1999, pg. 59-61. Tries cubic, quadratic, and bisection methods
    of zooming.
    """
    state = _ZoomState(
        done=False,
        failed=False,
        j=0,
        a_lo=a_lo,
        phi_lo=phi_lo,
        dphi_lo=dphi_lo,
        a_hi=a_hi,
        phi_hi=phi_hi,
        dphi_hi=dphi_hi,
        a_rec=(a_lo + a_hi) / 2.0,
        phi_rec=(phi_lo + phi_hi) / 2.0,
        a_star=1.0,
        phi_star=phi_lo,
        dphi_star=dphi_lo,
        g_star=g_0,
        nfev=0,
        ngev=0,
    )
    delta1 = 0.2
    delta2 = 0.1

    def body(state):
        # Body of zoom algorithm. We use boolean arithmetic to avoid using jax.cond
        # so that it works on GPU/TPU.
        dalpha = state.a_hi - state.a_lo
        a = jnp.minimum(state.a_hi, state.a_lo)
        b = jnp.maximum(state.a_hi, state.a_lo)
        cchk = delta1 * dalpha
        qchk = delta2 * dalpha

        # This will cause the line search to stop, and since the Wolfe conditions
        # are not satisfied the minimization should stop too.
        threshold = jnp.where((bits < 64), 1e-5, 1e-10)  # jnp.finfo(dalpha).bits
        state = state._replace(failed=state.failed | (dalpha <= threshold))

        # Cubmin is sometimes nan, though in this case the bounds check will fail.
        a_j_cubic = _cubicmin(
            state.a_lo,
            state.phi_lo,
            state.dphi_lo,
            state.a_hi,
            state.phi_hi,
            state.a_rec,
            state.phi_rec,
        )
        use_cubic = (state.j > 0) & (a_j_cubic > a + cchk) & (a_j_cubic < b - cchk)
        a_j_quad = _quadmin(state.a_lo, state.phi_lo, state.dphi_lo, state.a_hi, state.phi_hi)
        use_quad = (~use_cubic) & (a_j_quad > a + qchk) & (a_j_quad < b - qchk)
        a_j_bisection = (state.a_lo + state.a_hi) / 2.0
        use_bisection = (~use_cubic) & (~use_quad)

        a_j = jnp.where(use_cubic, a_j_cubic, state.a_rec)
        a_j = jnp.where(use_quad, a_j_quad, a_j)
        a_j = jnp.where(use_bisection, a_j_bisection, a_j)

        # TODO(jakevdp): should we use some sort of fixed-point approach here instead?
        phi_j, dphi_j, g_j = restricted_func_and_grad(a_j)
        phi_j = phi_j.astype(state.phi_lo.dtype)
        dphi_j = dphi_j.astype(state.dphi_lo.dtype)
        g_j = g_j.astype(state.g_star.dtype)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)

        hi_to_j = wolfe_one_neg(a_j, phi_j) | (phi_j >= state.phi_lo)
        star_to_j = wolfe_two(dphi_j) & (~hi_to_j)
        hi_to_lo = (dphi_j * (state.a_hi - state.a_lo) >= 0.0) & (~hi_to_j) & (~star_to_j)
        lo_to_j = (~hi_to_j) & (~star_to_j)

        state = state._replace(
            **_binary_replace(
                hi_to_j,
                state._asdict(),
                dict(
                    a_hi=a_j,
                    phi_hi=phi_j,
                    dphi_hi=dphi_j,
                    a_rec=state.a_hi,
                    phi_rec=state.phi_hi,
                ),
            ),
        )

        # for termination
        state = state._replace(
            done=star_to_j | state.done,
            **_binary_replace(
                star_to_j,
                state._asdict(),
                dict(
                    a_star=a_j,
                    phi_star=phi_j,
                    dphi_star=dphi_j,
                    g_star=g_j,
                ),
            ),
        )
        state = state._replace(
            **_binary_replace(
                hi_to_lo,
                state._asdict(),
                dict(
                    a_hi=state.a_lo,
                    phi_hi=state.phi_lo,
                    dphi_hi=state.dphi_lo,
                    a_rec=state.a_hi,
                    phi_rec=state.phi_hi,
                ),
            ),
        )
        state = state._replace(
            **_binary_replace(
                lo_to_j,
                state._asdict(),
                dict(
                    a_lo=a_j,
                    phi_lo=phi_j,
                    dphi_lo=dphi_j,
                    a_rec=state.a_lo,
                    phi_rec=state.phi_lo,
                ),
            ),
        )
        state = state._replace(j=state.j + 1)
        # Choose higher cutoff for maxiter than Scipy as Jax takes longer to find
        # the same value - possibly floating point issues?
        state = state._replace(failed=state.failed | state.j >= maxiter_zoom)  # custom change
        return state

    state = lax.while_loop(
        lambda state: (~state.done) & (~pass_through) & (~state.failed), body, state
    )

    return state


class _LineSearchState(NamedTuple):
    done: Union[bool, jnp.ndarray]
    failed: Union[bool, jnp.ndarray]
    i: Union[int, jnp.ndarray]
    a_i1: Union[float, jnp.ndarray]
    phi_i1: Union[float, jnp.ndarray]
    dphi_i1: Union[float, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]
    a_star: Union[float, jnp.ndarray]
    phi_star: Union[float, jnp.ndarray]
    dphi_star: Union[float, jnp.ndarray]
    g_star: jnp.ndarray


class _LineSearchResults(NamedTuple):
    """Results of line search.

    Parameters:
      failed: True if the strong Wolfe criteria were satisfied
      nit: integer number of iterations
      nfev: integer number of functions evaluations
      ngev: integer number of gradients evaluations
      k: integer number of iterations
      a_k: integer step size
      f_k: final function value
      g_k: final gradient value
      status: integer end status
    """

    failed: Union[bool, jnp.ndarray]
    nit: Union[int, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]
    k: Union[int, jnp.ndarray]
    a_k: Union[int, jnp.ndarray]
    f_k: jnp.ndarray
    g_k: jnp.ndarray
    status: Union[bool, jnp.ndarray]


@jax.jit
def custom_line_search_jax(
    fun,
    args,
    xk,
    pk,
    old_fval=None,
    old_old_fval=None,
    gfk=None,
    c1=1e-4,  # 1e-4,
    c2=0.9,  # 0.2 (Solver time : 1332.60), #0.99 not working, #0.5 (Solver time : 1127.87), #0.9 (Solver time : 1159.39),
    maxiter_main=20,  # 200  Solver time : 1204.75
    maxiter_zoom=30,
):
    """Inexact line search that satisfies strong Wolfe conditions.

    Algorithm 3.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-61

    Args:
      fun: function of the form f(x) where x is a flat ndarray and returns a real
        scalar. The function should be composed of operations with vjp defined.
      x0: initial guess.
      pk: direction to search in. Assumes the direction is a descent direction.
      old_fval, gfk: initial value of value_and_gradient as position.
      old_old_fval: unused argument, only for scipy API compliance.
      maxiter: maximum number of iterations to search
      c1, c2: Wolfe criteria constant, see ref.

    Returns: LineSearchResults
    """

    def restricted_func_and_grad(t):
        phi, g = jax.value_and_grad(fun)(xk + t * pk, args)  ###

        dphi = jnp.real(_dot(g, pk))
        return phi, dphi, g

    if old_fval is None or gfk is None:
        phi_0, dphi_0, gfk = restricted_func_and_grad(0.0)
    else:
        phi_0 = old_fval
        dphi_0 = jnp.real(_dot(gfk, pk))
    if old_old_fval is not None:
        candidate_start_value = 1.01 * 2 * (phi_0 - old_old_fval) / dphi_0
        start_value = jnp.where(candidate_start_value > 1, 1.0, candidate_start_value)
    else:
        start_value = 1

    # Hager Zhang:
    # https://www.math.lsu.edu/~hozhang/papers/cg_descent.pdf
    # def wolfe_approx_one(dphi_i):
    #     delta = 0.1
    #     return (2 * delta - 1) * dphi_0 >= dphi_i

    # def wolfe_approx_fun(phi_i):
    #     eps = 1e-6
    #     return phi_i <= (1 + eps) * jnp.abs(phi_0)

    # def _wolfe_one(a_i, phi_i):
    #     return phi_i <= phi_0 + c1 * a_i * dphi_0  # numerical problems

    # def wolfe_one_neg_hz(a_i, phi_i, dphi_i):
    #     return ~(_wolfe_one(a_i, phi_i) | (wolfe_approx_fun(phi_i) & wolfe_approx_one(dphi_i)))

    def wolfe_one_neg(a_i, phi_i):
        return phi_i > phi_0 + c1 * a_i * dphi_0
        # return ~(_wolfe_one(a_i, phi_i) | (wolfe_approx_fun(phi_i) & wolfe_approx_one(dphi_i)))

    def wolfe_two(dphi_i):
        # return jnp.abs(dphi_i) <= -c2 * dphi_0

        # return jnp.abs(dphi_i) <= jnp.abs(c2 * dphi_0)
        # page 75: We assume that pk is a descent direction —
        # that is, \dphi(0) < 0 — so that oursearch can be confined to positive values of α.

        # Customization: weak Wolfe condition
        return dphi_i >= c2 * dphi_0

    state = _LineSearchState(
        done=False,
        failed=False,
        # algorithm begins at 1 as per Wright and Nocedal, however Scipy has a
        # bug and starts at 0. See https://github.com/scipy/scipy/issues/12157
        i=1,
        a_i1=0.0,
        phi_i1=phi_0,
        dphi_i1=dphi_0,
        nfev=1 if (old_fval is None or gfk is None) else 0,
        ngev=1 if (old_fval is None or gfk is None) else 0,
        a_star=0.0,
        phi_star=phi_0,
        dphi_star=dphi_0,
        g_star=gfk,
    )

    def body(state):
        # no amax in this version, we just double as in scipy.
        # unlike original algorithm we do our next choice at the start of this loop
        a_i = jnp.where(state.i == 1, start_value, state.a_i1 * 2.0)

        phi_i, dphi_i, g_i = restricted_func_and_grad(a_i)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)

        star_to_zoom1 = wolfe_one_neg(a_i, phi_i) | ((phi_i >= state.phi_i1) & (state.i > 1))
        star_to_i = wolfe_two(dphi_i) & (~star_to_zoom1)
        star_to_zoom2 = (dphi_i >= 0.0) & (~star_to_zoom1) & (~star_to_i)

        zoom1 = _zoom(
            restricted_func_and_grad,
            wolfe_one_neg,
            wolfe_two,
            state.a_i1,
            state.phi_i1,
            state.dphi_i1,
            a_i,
            phi_i,
            dphi_i,
            gfk,
            ~star_to_zoom1,
            maxiter_zoom,
        )

        state = state._replace(nfev=state.nfev + zoom1.nfev, ngev=state.ngev + zoom1.ngev)

        zoom2 = _zoom(
            restricted_func_and_grad,
            wolfe_one_neg,
            wolfe_two,
            a_i,
            phi_i,
            dphi_i,
            state.a_i1,
            state.phi_i1,
            state.dphi_i1,
            gfk,
            ~star_to_zoom2,
            maxiter_zoom,
        )

        state = state._replace(nfev=state.nfev + zoom2.nfev, ngev=state.ngev + zoom2.ngev)

        state = state._replace(
            done=star_to_zoom1 | state.done,
            failed=(star_to_zoom1 & zoom1.failed) | state.failed,
            **_binary_replace(
                star_to_zoom1,
                state._asdict(),
                zoom1._asdict(),
                keys=["a_star", "phi_star", "dphi_star", "g_star"],
            ),
        )
        state = state._replace(
            done=star_to_i | state.done,
            **_binary_replace(
                star_to_i,
                state._asdict(),
                dict(
                    a_star=a_i,
                    phi_star=phi_i,
                    dphi_star=dphi_i,
                    g_star=g_i,
                ),
            ),
        )
        state = state._replace(
            done=star_to_zoom2 | state.done,
            failed=(star_to_zoom2 & zoom2.failed) | state.failed,
            **_binary_replace(
                star_to_zoom2,
                state._asdict(),
                zoom2._asdict(),
                keys=["a_star", "phi_star", "dphi_star", "g_star"],
            ),
        )
        state = state._replace(i=state.i + 1, a_i1=a_i, phi_i1=phi_i, dphi_i1=dphi_i)
        return state

    state = lax.while_loop(
        lambda state: (~state.done) & (state.i <= maxiter_main) & (~state.failed), body, state
    )

    status = jnp.where(
        state.failed,
        jnp.array(1),  # zoom failed
        jnp.where(
            state.i > maxiter_main,
            jnp.array(3),  # maxiter reached
            jnp.array(0),  # passed (should be)
        ),
    )
    # Step sizes which are too small causes the optimizer to get stuck with a
    # direction of zero in <64 bit mode - avoid with a floor on minimum step size.
    alpha_k = state.a_star
    alpha_k = jnp.where(
        (bits != 64) & (jnp.abs(alpha_k) < 1e-8),  # jnp.finfo(alpha_k).bits
        jnp.sign(alpha_k) * 1e-8,
        alpha_k,
    )
    results = _LineSearchResults(
        failed=state.failed | (~state.done),
        nit=state.i - 1,  # because iterations started at 1
        nfev=state.nfev,
        ngev=state.ngev,
        k=state.i,
        a_k=alpha_k,
        f_k=state.phi_star,
        g_k=state.g_star,
        status=status,
    )
    return results


##############################

Array = Any


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
    failed: Union[bool, Array]
    k: Union[int, Array]
    nfev: Union[int, Array]
    ngev: Union[int, Array]
    x_k: Array
    f_k: Array
    g_k: Array
    s_history: Array
    y_history: Array
    rho_history: Array
    history_position: int
    gamma: Union[float, Array]
    status: Union[int, Array]
    ls_status: Union[int, Array]
    ###
    fun: Callable
    args: dict
    hes_inv: float
    # xtol_max: float
    # xtol_mean: float
    maxiter_main_ls: float
    maxiter_zoom_ls: float
    ###
    maxiter: float
    norm: float
    ftol: float
    gtol: float
    maxfun: float
    maxgrad: float
    # maxls: float
    xtol: float

    # Added custom rolling buffer
    def get_s_history(self, i):
        return self.s_history[(i + self.history_position) % len(self.s_history)]

    def get_y_history(self, i):
        return self.y_history[(i + self.history_position) % len(self.y_history)]

    def get_rho_history(self, i):
        return self.rho_history[(i + self.history_position) % len(self.rho_history)]

    def get_updated_history_position(self, history):
        # return 0
        return (self.history_position + 1) % len(history)

    def get_updated_history(self, history, new):
        # return jnp.roll(history, -1, axis=0).at[-1, ...].set(new)
        return history.at[self.history_position, ...].set(new)


def get_backend():
    key = "OPTIMIZATION_BACKEND"
    return os.environ[key] if key in os.environ else None


def get_state_initial(
    fun,
    f_0,
    g_0,
    hes_inv,
    args,
    x0: Array,
    norm=jnp.inf,
    maxcor: int = 10,
    ftol: float = 2.220446049250313e-09,
    gtol: float = 1e-05,
    maxfun: Optional[float] = None,
    maxgrad: Optional[float] = None,
    xtol: float = None,
    maxiter: Optional[float] = None,  # 200,  # None
    maxiter_main_ls: int = 20,
    maxiter_zoom_ls: int = 30,
):
    # print("Minimize")
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

    # ensure there is at least one termination condition
    # if (maxiter is None) and (maxfun is None) and (maxgrad is None):
    #     maxiter = d * 200

    # set others to inf, such that >= is supported
    if maxiter is None:
        maxiter = jnp.inf
    if maxfun is None:
        maxfun = jnp.inf
    if maxgrad is None:
        maxgrad = jnp.inf

    state_initial = LBFGSResults(
        converged=False,
        failed=False,
        k=0,
        nfev=1,
        ngev=1,
        x_k=x0,
        f_k=f_0,
        g_k=g_0,
        s_history=jnp.zeros(
            (maxcor, d), dtype=dtype
        ),  # if opti_state is None else opti_state.s_history,
        y_history=jnp.zeros(
            (maxcor, d), dtype=dtype
        ),  # if opti_state is None else opti_state.y_history,
        history_position=-1,  # opti_state...
        rho_history=jnp.zeros(
            (maxcor,), dtype=dtype
        ),  # if opti_state is None else opti_state.rho_history,
        gamma=1.0,
        status=0,
        ls_status=0,
        ###
        fun=jax.tree_util.Partial(fun),
        args=args,
        hes_inv=hes_inv,  # jax.tree_util.Partial(opti_state),  # if hes is None else jax.tree_util.Partial(hes),
        # xtol_max=xtol_max,
        # xtol_mean=xtol_mean,
        # max_iter=max_iter,
        maxiter=maxiter,
        norm=norm,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxgrad=maxgrad,
        maxiter_main_ls=maxiter_main_ls,
        maxiter_zoom_ls=maxiter_zoom_ls,
        xtol=xtol,
    )
    return state_initial


def get_opti_fun(state_initial):
    return jax.jit(opti_fun, backend=get_backend()).lower(state_initial).compile()


def opti_fun(state):
    return lax.while_loop(cond_fun_jax, body_fun_jax, state)


def _two_loop_recursion(state: LBFGSResults):
    his_size = len(state.rho_history)
    curr_size = jnp.where(state.k < his_size, state.k, his_size)
    q = -jnp.conj(state.g_k)
    a_his = jnp.zeros_like(state.rho_history)

    def body_fun1(j, carry):
        i = his_size - 1 - j
        _q, _a_his = carry
        a_i = state.get_rho_history(i) * jnp.real(_dot(jnp.conj(state.get_s_history(i)), _q))
        _a_his = _a_his.at[i].set(a_i)
        _q = _q - a_i * jnp.conj(state.get_y_history(i))
        return _q, _a_his

    q, a_his = lax.fori_loop(0, curr_size, body_fun1, (q, a_his))

    # Added custom Hessian preconditioning
    if state.hes_inv is None:
        r = state.gamma * q
        # r = q
    else:
        r = state.hes_inv @ q

    def body_fun2(j, _r):
        i = his_size - curr_size + j
        b_i = state.get_rho_history(i) * jnp.real(_dot(state.get_y_history(i), _r))
        _r = _r + (a_his[i] - b_i) * state.get_s_history(i)
        return _r

    r = lax.fori_loop(0, curr_size, body_fun2, r)
    return r


def cond_fun_jax(state: LBFGSResults):
    return (~state.converged) & (~state.failed)


def body_fun_jax(state: LBFGSResults):
    # find search direction
    p_k = _two_loop_recursion(state)

    # line search
    ls_results = custom_line_search_jax(
        fun=state.fun,
        args=state.args,  ###
        xk=state.x_k,
        pk=p_k,
        old_fval=state.f_k,
        gfk=state.g_k,
        maxiter_main=state.maxiter_main_ls,
        maxiter_zoom=state.maxiter_zoom_ls,
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

    # replacements for next iteration
    status = 0
    status = jnp.where(ls_results.failed, 5, status)
    # status = jnp.where(state.f_k - f_kp1 < state.ftol, 4, status)
    # status = jnp.where(state.ngev >= state.maxgrad, 3, status)  # type: ignore
    # status = jnp.where(state.nfev >= state.maxfun, 2, status)  # type: ignore
    status = jnp.where(state.k >= state.maxiter, 1, status)  # type: ignore

    # Added custom stopping criterion
    # norm = 2  # jnp.inf  # state.norm ###
    # converged = jnp.linalg.norm(g_kp1, ord=norm) < state.gtol
    # converged = jnp.linalg.norm(s_k, ord=norm) < state.xtol

    converged = state.f_k - f_kp1 <= state.ftol
    # scipy: `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.

    # TODO(jakevdp): use a fixed-point procedure rather than type-casting?
    state = state._replace(
        converged=converged,
        failed=(status > 0) & (~converged),
        k=state.k + 1,
        nfev=state.nfev + ls_results.nfev,
        ngev=state.ngev + ls_results.ngev,
        x_k=x_kp1.astype(state.x_k.dtype),
        f_k=f_kp1.astype(state.f_k.dtype),
        g_k=g_kp1.astype(state.g_k.dtype),
        s_history=state.get_updated_history(history=state.s_history, new=s_k),
        y_history=state.get_updated_history(history=state.y_history, new=y_k),
        rho_history=state.get_updated_history(history=state.rho_history, new=rho_k),
        history_position=state.get_updated_history_position(history=state.s_history),
        gamma=gamma,
        status=jnp.where(converged, 0, status),
        ls_status=ls_results.status,
    )

    return state
