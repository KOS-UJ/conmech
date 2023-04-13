# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman <piotr.bartman@uj.edu.pl>
# Copyright (C) 2023  Adil Bagirov <a.bagirov@federation.edu.au>
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
import numba
import numpy as np

"""
The code of (Limited memory) Quasi secant method

nbundle - is the size of bundle, the maximum number of
subgradients or discrete gradients which can be
calculated at each iteration

ngradient = False if you use approximation to subgradient
ngradient = True if you use exact gradients of the objective
              and constraint functions
"""


@numba.njit()
def gradient(x):
    raise NotImplementedError()


# ======================================================================
# Without scaling
# ======================================================================
# def qsmlm(nvar,x0,nbundle,ngradient,x,f,cputime,finit):
#     # implicit double precision (a-h,o-z)
#     # PARAMETER(maxvar=2000)
#     # double precision x(maxvar),x0(maxvar)
#     # COMMON /csize/m,/citer/maxiter,niter,/cnf/nf,/cgrad/ngradient1
#     #      1 ,/cngrad2/ngrad2
#     # call cpu_time(time1)
#     m=nvar
#     nf=0
#     niter=0
#     ngrad2=0
#     nfgrad=0
#     ngradient1=ngradient
#     maxiter=10000
#     slinit=1.0
#     for i in range(m):
#         x[i] = x0[i]
#     f = fv(x)
#     finit=f
#     x = optimum(x,nbundle,slinit)
#     f = fv(x)
#     # call cpu_time(time2)
#     # cputime=time2-time1
#     #===========================================================
#     return
# ============================================================


def minimize(loss, x, args, nbundle=None, slinit=1, ngradient=False, maxiter=1000):
    """
    Find minimum of `loss` function.
    """
    nbundle = min(200, 2 * len(x) + 3) if not nbundle else nbundle
    step0 = -2.0e-01
    div = 5.0e-01
    eps0 = 1.0e-07
    slmin = 1.0e-10 * slinit
    pwt = 1.0e-08
    sdif = 1.0e-05
    mturn = 3
    vbar = np.empty_like(x)
    g = np.empty_like(x)
    x1 = np.empty_like(x)
    tildev = np.empty_like(x)
    fvalues = np.empty(maxiter)
    f4 = 0.
    m = len(x)

    sl = slinit / div
    f2 = loss(x, *args)[0]
    niter = 0
    while True:
        sl = div * sl
        if sl < slmin:
            break
        for i in range(m):
            g[i] = 1.0e+00 / np.sqrt(m)
        nnew = 0
        outer_break_flag = False

        while True:
            niter = niter + 1
            if niter >= maxiter:
                outer_break_flag = True
                break

            nnew = nnew + 1
            f1 = f2
            fvalues[niter] = f1

            if nnew > mturn:
                mturn2 = niter - mturn + 1
                ratio1 = (fvalues[mturn2] - f1) / (abs(f1) + 1.0e+00)
                if ratio1 < sdif:
                    break

            if nnew >= (2 * mturn):
                mturn2 = niter - 2 * mturn + 1
                ratio1 = (fvalues[mturn2] - f1) / (abs(f1) + 1.0e+00)
                if ratio1 < (1.0e+01 * sdif):
                    break

            break_flag = False
            for ndg in range(nbundle):
                v = dgrad(loss, x, args, sl, g, f4, ndg, pwt, ngradient)
                dotprod = 0.0e+00
                for i in range(m):
                    dotprod = dotprod + v[i] * v[i]

                r = np.sqrt(dotprod)
                if r < eps0:
                    break_flag = True
                    break

                if ndg == 0:
                    for i in range(m):
                        tildev[i] = v[i]

                clambda = wolfe(tildev, v, vbar)
                if ndg > 0 and clambda <= 1.0e-04:
                    break_flag = True
                    break

                r = 0
                for i in range(m):
                    r = r + vbar[i] ** 2

                r = np.sqrt(r)
                if r < eps0:
                    break_flag = True
                    break

                for i in range(m):
                    g[i] = -vbar[i] / r
                    x1[i] = x[i] + sl * g[i]

                f4 = loss(x1, *args)[0]
                f3 = (f4 - f1) / sl
                decreas = step0 * r
                if f3 < decreas:
                    step, f5 = armijo(loss, x, args, g, f1, f4, sl, r)
                    f2 = f5
                    for i in range(m):
                        x[i] = x[i] + step * g[i]
                    sl = 1.2e+00 * sl
                    break
                for i in range(m):
                    tildev[i] = vbar[i]

            if break_flag:
                break
        if outer_break_flag:
            break
    return x


@numba.njit()
def wolfe(tildev: np.ndarray, v: np.ndarray, vbar: np.ndarray):
    """
    Solves quadratic programming problem, to find descent direction,
    Step 3, Algorithm 2.
    """
    r1 = 0.0e+00
    r2 = 0.0e+00
    m = len(tildev)
    for i in range(m):
        r1 += tildev[i] * (tildev[i] - v[i])
        r2 += (tildev[i] - v[i]) * (tildev[i] - v[i])
    if r2 <= 1.0e-04:
        clambda = 0.0e+00
    else:
        clambda = r1 / r2
        if clambda < 0.0:
            clambda = 0.0
        if clambda > 1.0:
            clambda = 1.0

    for i in range(m):
        vbar[i] = tildev[i] + clambda * (v[i] - tildev[i])
    return clambda  # TODO


@numba.njit()
def dgrad(loss, x, args, sl, g, f4, ndg, pwt, ngradient):
    """
    Calculates subgradients or discrete gradients
    """

    x1 = np.empty_like(x)
    for k in range(len(x1)):
        x1[k] = x[k] + sl * g[k]

    if ngradient:
        raise NotImplementedError()
        # dg = gradient(x1)
    else:
        if ndg > 0:
            r2 = f4
        else:
            r2 = loss(x1, *args)[0]
        dg = dgrad2(loss, x1, args, r2, pwt)
        # ngrad2=ngrad2+1

    return dg


@numba.njit()
def dgrad2(loss, x1, args, r2, pwt):
    """
    Calculates discrete gradients: Step 5, Algorithm 1
    """
    alpha = 1.0e+00
    t = pwt / alpha
    r4 = r2
    v = np.empty_like(x1)
    for k in range(len(x1)):
        t = t * alpha
        r3 = r4
        x1[k] = x1[k] + t
        r4 = loss(x1, *args)[0]
        # nfgrad=nfgrad+1
        v[k] = (r4 - r3) / t
    return v


@numba.njit()
def armijo(loss, x, args, g, f1, f4, sl, r):
    """
    Line search (Armijo-type), Step 5 Algorithm 2.
    """
    step = sl
    f5 = f4
    step1 = sl
    k = 0
    x1 = np.empty_like(x)
    while True:
        k = k + 1
        if k > 20:
            break
        step1 = 2.0 * step1
        for i in range(len(x)):
            x1[i] = x[i] + step1 * g[i]
        f50 = loss(x1, *args)[0]
        f30 = f50 - f1 + 5.0e-02 * step1 * r
        if f30 > 0.0:
            break
        step = step1
        f5 = f50
    return step, f5
