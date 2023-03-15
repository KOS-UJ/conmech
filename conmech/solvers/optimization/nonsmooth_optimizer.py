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
import numpy as np

#     Last change:  LJ    9 Sep 2011   10:39 am

# The code of (Limited memory) Quasi secant method
#       implicit double precision (a-h,o-z)
#       PARAMETER(maxvar=2000)
#       double precision x0(maxvar),x(maxvar)
#       COMMON /citer/maxiter,niter,/cnf/nf,/cngrad2/ngrad2,/cka/ka(500)
#       character*30 outfi/'results.txt'/
#       open(40,file=outfi)
# =======================================================
#  Input data:
#  n        - number of variables
# =======================================================
# n = 5
# ======================================================
# nbundle - is the size of bundle, the maximum number of
# subgradients or discrete gradients which can be
# calculated at each iteration
# ======================================================
# nbundle = min(200, 2 * n + 3)
# ======================================================
# ngradient = 0 if you use approximation to subgradient
# ngradient = 1 if you use exact gradients of the objective
#               and constraint functions
# ======================================================
# ngradient = True
# =======================================================
#  Starting point
# =======================================================
# x0 = np.zeros(n)


# =======================================================
#  Calling QSM (create output file)
# =======================================================
#     call qsmlm(n,x0,nbundle,ngradient,x,fvalue,cputime,finit)
#     WRITE(40,41) finit,fvalue,niter,nf,ngrad2,cputime
# 41  FORMAT(2f16.8,3i8,f10.4)
# =======================================================
# CLOSE(40)
# CLOSE(81)
# STOP
# END
# ======================================================================
# def func(x,objf):
# implicit double precision (a-h,o-z)
# PARAMETER(maxvar=2000)
# double precision x(maxvar),ff1(500),ff2(500),ff(500)
# COMMON /cka/ka(500)
# ===============================================================
# summ=0.0
# for i in range(0,500):  # do i=1,499
#     ff1(i)=-x(i)-x(i+1)
#     ff2(i)=-x(i)-x(i+1)+x(i)**2+x(i+1)**2-1.0d+00
#     ff(i)=dmax1(ff1(i),ff2(i))
#     summ=summ+ff(i)
# objf=summ
#
# for i in range(0,500):  # do i=1,499
#     ka(i)=0.0d+00
#     IF(ff(i).eq.ff1(i)) ka(i)=1
#
# return
# ======================================================================
def gradient(x):
    # implicit double precision (a-h,o-z)
    # PARAMETER(maxvar=2000)
    # double precision x(maxvar),grad(maxvar)
    # COMMON /cka/ka(500)
    # ===============================================================
    # if (ka(1).eq.1) then
    #   grad(1)=-1.0d+00
    # end if
    # if (ka(1).eq.0) then
    #   grad(1)=-1.0d+00+2.0d+00*x(1)
    # end if
    #
    # do i=2,499
    #   grad(i)=ka(i-1)*(-1.0d+00)+(1-ka(i-1))*(-1.0d+00+2*x(i))
    #      1   +ka(i)*(-1.0d+00)+(1-ka(i))*(-1.0d+00+2*x(i))
    # end do
    #
    #
    # if (ka(499).eq.1) then
    #   grad(500)=-1.0d+00
    # end if
    # if (ka(499).eq.0) then
    #   grad(500)=-1.0d+00+2.0d+00*x(500)
    # end if
    # ===============================================================
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


def optimum(loss, x, args, nbundle=None, slinit=1, ngradient=True):
    """
    From Fortran:
    implicit double precision (a-h,o-z)
    PARAMETER(maxvar=2000, maxdg=1000,maxit=100000)
    double precision x(maxvar),x1(maxvar),g(maxvar),v(maxvar)
         1 ,fvalues(maxit),tildev(maxvar),vbar(maxvar)
    common /csize/m,/citer/maxiter,niter
    """
    nbundle = min(200, 2 * len(x) + 3) if not nbundle else nbundle
    step0 = -2.0e-01
    div = 5.0e-01
    eps0 = 1.0e-07
    slmin = 1.0e-10 * slinit
    pwt = 1.0e-08
    sdif = 1.0e-05
    mturn = 3
    maxiter = 1000  # TODO
    maxit = 100000
    vbar = np.empty_like(x)
    g = np.empty_like(x)
    x1 = np.empty_like(x)
    tildev = np.empty_like(x)
    fvalues = np.empty(maxit)  # TODO
    f4 = None
    m = len(x)

    sl = slinit / div
    f2 = loss(x, *args)
    niter = 0
    while True:
        sl = div * sl
        if sl < slmin:
            break
        for i in range(m):
            g[i] = 1.0e+00 / np.sqrt(m)
        nnew = 0
        # ================================================================
        outer_break_flag = False
        while True:
            niter = niter + 1
            if niter > maxiter:
                outer_break_flag = True
                break
            nnew = nnew + 1
            f1 = f2
            fvalues[niter] = f1
            # ---------------------------------------------------------------
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
            # --------------------------------------------------------------
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

                f4 = loss(x1, *args)
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


# ==============================================================
#  Subroutines Wolfe and Equations solves quadratic
#  programming problem, to find
#  descent direction, Step 3, Algorithm 2.
# ===============================================================

def wolfe(tildev, v, vbar):
    # implicit double precision (a-h,o-z)
    # PARAMETER(maxvar=2000, maxdg=1000)
    # common /csize/m
    # double precision tildev(maxvar), v(maxvar), vbar(maxvar)
    # =============================================================
    r1 = 0.0e+00
    r2 = 0.0e+00
    m = len(tildev)
    for i in range(m):
        r1 = r1 + tildev(i) * (tildev(i) - v(i))
        r2 = r2 + (tildev(i) - v(i)) * (tildev(i) - v(i))
    if r2 <= 1.0e-04:
        clambda = 0.0e+00
    else:
        clambda = r1 / r2
        if clambda < 0.0:
            clambda = 0.0
        if clambda > 1.0:
            clambda = 1.0

    for i in range(m):
        vbar[i] = tildev(i) + clambda * (v(i) - tildev(i))
    return clambda  # TODO


# =====================================================================
# Subroutine dgrad calculates subgradients or discrete gradients
# =====================================================================
def dgrad(loss, x, args, sl, g, f4, ndg, pwt, ngradient):
    # implicit double precision (a-h,o-z)
    # PARAMETER(maxvar=2000, maxdg=1000)
    # double precision x1(maxvar),g(maxvar),x(maxvar),dg(maxvar)
    # common /csize/m,/cgrad/ngradient1,/cngrad2/ngrad2

    x1 = np.empty_like(x)
    for k in range(len(x1)):
        x1[k] = x[k] + sl * g[k]

    if ngradient:
        dg = gradient(x1)
    else:
        if ndg > 0:
            r2 = f4
        else:
            r2 = loss(x1, *args)
        dg = dgrad2(loss, x1, args, r2, pwt)
        # ngrad2=ngrad2+1

    return dg


# =====================================================================
# Subroutine dgrad calculates discrete gradients: Step 5, Algorithm 1
# =====================================================================
def dgrad2(loss, x1, args, r2, pwt):
    # implicit double precision (a-h,o-z)
    # PARAMETER(maxvar=2000, maxdg=1000)
    # double precision x1(maxvar),v(maxvar)
    # common /csize/m,/cngrad2/ngrad2
    alpha = 1.0e+00
    t = pwt / alpha
    r4 = r2
    v = np.empty_like(x1)
    for k in range(len(x1)):
        t = t * alpha
        r3 = r4
        x1[k] = x1[k] + t
        r4 = loss(x1, *args)
        # nfgrad=nfgrad+1
        v[k] = (r4 - r3) / t
    return v


# ===========================================================
# Line search (Armijo-type), Step 5 Algorithm 2.
# ===========================================================
def armijo(loss, x, args, g, f1, f4, sl, r):
    # implicit double precision (a-h,o-z)
    # PARAMETER(maxvar=2000, maxdg=1000)
    # common /csize/m
    # double precision x(maxvar),g(maxvar),x1(maxvar)
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
        f50 = loss(x1, *args)
        f30 = f50 - f1 + 5.0e-02 * step1 * r
        if f30 > 0.0:
            break
        step = step1
        f5 = f50
    return step, f5


# def fv(x, *args):
#     # implicit double precision (a-h,o-z)
#     # PARAMETER(maxvar=2000)
#     # double precision x(maxvar)
#     # COMMON /cnf/nf,/csize/m
#     # ================================================================
#     # call func(x,objf)
#     # nf=nf+1
#     # f=objf
#     # =================================================================
#     return f(x, args)
