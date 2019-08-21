"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np
from simulation.matrices import Matrices
from simulation.f import F
import pylab


class Solver:

    def __init__(self, setting, timeStep, F0, FN, mi, la):

        self.mi = mi
        self.la = la

        self.s = setting
        self.tS = timeStep
        self.currentTime = 0

        self.M = Matrices(setting, mi, la)
        self.F = F(setting, F0, FN)

        self.u = np.zeros([self.s.indNumber(), 2])

        self.DisplacedPoints = np.zeros([len(self.s.Points), 3])

        for i in range(0, len(self.s.Points)):
            self.DisplacedPoints[i] = self.s.Points[i]

    def norm(self, v):
        return pylab.sqrt(v[0] * v[0] + v[1] * v[1])

    def length(self, p1, p2):
        return float(pylab.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))

    def nDown(self, e):
        # [0,-1]
        e1 = int(self.s.Edges[e][0])
        e2 = int(self.s.Edges[e][1])
        dx = self.s.Points[e2][0] - self.s.Points[e1][0]
        dy = self.s.Points[e2][1] - self.s.Points[e1][1]
        norm = pylab.sqrt(dx * dx + dy * dy)
        n = np.array([float(dy) / norm, float(-dx) / norm])
        if (n[1] > 0):
            n = -n
        return n

    def nDownAtContactBoundary(self):
        N = np.zeros([self.s.indNumber(), 2])

        for i in range(0, self.s.indNumber()):
            for e in range(-self.s.BorderEdgesD - self.s.BorderEdgesN - self.s.BorderEdgesC,
                           -self.s.BorderEdgesD - self.s.BorderEdgesN):
                e1 = int(self.s.Edges[e][0])
                e2 = int(self.s.Edges[e][1])
                if (i == e1 or i == e2):
                    n = self.nDown(e)
                    if (N[i][0] == 0 and N[i][1] == 0):
                        N[i] = n
                    else:
                        N[i] = 0.5 * N[i] + 0.5 * n
        return N

    def JZu(self):
        JZu = np.zeros([self.s.indNumber(), 2])

        for i in range(0, self.s.indNumber()):
            for e in range(-self.s.BorderEdgesD - self.s.BorderEdgesN - self.s.BorderEdgesC,
                           -self.s.BorderEdgesD - self.s.BorderEdgesN):
                e1 = int(self.s.Edges[e][0])
                e2 = int(self.s.Edges[e][1])
                if (i == e1 or i == e2):
                    umL = 0  # u at mL
                    if (e1 < self.s.indNumber()):
                        umL += self.u[e1] * 0.5
                    if (e2 < self.s.indNumber()):
                        umL += self.u[e2] * 0.5

                    p1 = self.s.Points[int(e1)][0:2]
                    p2 = self.s.Points[int(e2)][0:2]
                    mL = (p1 + p2) * 0.5
                    L = self.length(p1, p2)
                    nmL = self.nDown(e)  # n at mL

                    uNmL = umL[0] * nmL[0] + umL[1] * nmL[1]
                    uTmL = umL - uNmL * nmL

                    vNZero = nmL[0]
                    vNOne = nmL[1]
                    vThauZero = [1. - float(nmL[0] * nmL[0]), - float(nmL[0] * nmL[1])]
                    vThauOne = [- float(nmL[0] * nmL[1]), 1. - float(nmL[1] * nmL[1])]

                    JZu[i][0] += L * 0.5 * (self.jnZ(uNmL, vNZero) + self.h(uNmL) * self.jtZ(uTmL, vThauZero))
                    JZu[i][1] += L * 0.5 * (self.jnZ(uNmL, vNOne) + self.h(uNmL) * self.jtZ(uTmL, vThauOne))

        return JZu

    def iterate(self, uVector):
        self.u[:, 0] = uVector[0:self.s.indNumber()]
        self.u[:, 1] = uVector[self.s.indNumber():2 * self.s.indNumber()]

        for i in range(0, self.s.indNumber()):
            self.DisplacedPoints[i][0] = self.s.Points[i][0] + self.u[i][0]
            self.DisplacedPoints[i][1] = self.s.Points[i][1] + self.u[i][1]

    def Bu1(self):
        return (np.dot(self.M.B11, self.u[:, 0]) + np.dot(self.M.B12, self.u[:, 1]))

    def Bu2(self):
        return (np.dot(self.M.B21, self.u[:, 0]) + np.dot(self.M.B22, self.u[:, 1]))

    def f(self, uVector):
        self.u = np.zeros([self.s.indNumber(), 2])
        self.u[:, 0] = uVector[0:self.s.indNumber()]
        self.u[:, 1] = uVector[self.s.indNumber():2 * self.s.indNumber()]

        X = self.Bu1() \
            + self.JZu()[:, 0] \
            - self.F.Zero

        Y = self.Bu2() \
            + self.JZu()[:, 1] \
            - self.F.One

        return 100000000 * np.append(X, Y)  # 10000000000

    ########################################################

    knu = 1.
    delta = 0.1

    def jnZ(self, uN, vN):  # un, vN - scalars
        # return 0
        if (uN <= 0):
            return 0 * vN
        return (self.knu * uN) * vN

    def h(self, uN):
        return 0
        # if(uN<=0):
        #    return 0
        # return 8.*uN

    def jtZ(self, uT, vT):  # uT, vT - vectors; REGULARYZACJA Coulomba
        rho = 0.0000001
        M = 1 / pylab.math.sqrt(float(uT[0] * uT[0] + uT[1] * uT[1]) + float(rho * rho))
        return M * float(uT[0]) * float(vT[0]) + M * float(uT[1]) * float(vT[1])

