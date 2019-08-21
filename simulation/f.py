"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np

class F:

    def __init__(self, setting, F0, FN):
        self.F0 = F0
        self.FN = FN
        self.s = setting
        self.F = np.zeros([self.s.indNumber(), 2])
        self.Zero = np.zeros([self.s.indNumber()])
        self.One = np.zeros([self.s.indNumber()])

    ########################################################

    def f0(self, x):
        return self.F0

    def fN(self, x):
        return self.FN

    ########################################################

    def setF(self):
        halfLongTriangleSide = self.s.halfLongTriangleSide
        halfShortTriangleSide = self.s.halfShortTriangleSide

        self.F = np.zeros([self.s.indNumber(), 2])

        for i in range(0, self.s.indNumber()):
            x = self.s.Points[i][0]
            y = self.s.Points[i][1]
            t = self.s.Points[i][2]

            if (t != 9):  # normal point

                valuesInTriangle = np.zeros([8, 2])

                valuesInTriangle[0] += self.f0([x - halfLongTriangleSide, y])
                valuesInTriangle[0] += self.f0([x - halfShortTriangleSide, y + halfShortTriangleSide])

                valuesInTriangle[1] += self.f0([x - halfShortTriangleSide, y + halfShortTriangleSide])
                valuesInTriangle[1] += self.f0([x, y + halfLongTriangleSide])

                valuesInTriangle[2] += self.f0([x, y + halfLongTriangleSide])
                valuesInTriangle[2] += self.f0([x + halfShortTriangleSide, y + halfShortTriangleSide])

                valuesInTriangle[3] += self.f0([x + halfShortTriangleSide, y + halfShortTriangleSide])
                valuesInTriangle[3] += self.f0([x + halfLongTriangleSide, y])

                valuesInTriangle[4] += self.f0([x + halfLongTriangleSide, y])
                valuesInTriangle[4] += self.f0([x + halfShortTriangleSide, y - halfShortTriangleSide])

                valuesInTriangle[5] += self.f0([x + halfShortTriangleSide, y - halfShortTriangleSide])
                valuesInTriangle[5] += self.f0([x, y - halfLongTriangleSide])

                valuesInTriangle[6] += self.f0([x, y - halfLongTriangleSide])
                valuesInTriangle[6] += self.f0([x - halfShortTriangleSide, y - halfShortTriangleSide])

                valuesInTriangle[7] += self.f0([x - halfShortTriangleSide, y - halfShortTriangleSide])
                valuesInTriangle[7] += self.f0([x - halfLongTriangleSide, y])

                if (t == 3):  # 3 - top
                    self.F[i] += valuesInTriangle[4]
                    self.F[i] += valuesInTriangle[5]
                    self.F[i] += valuesInTriangle[6]
                    self.F[i] += valuesInTriangle[7]
                if (t == 4):  # 4 - right top corner
                    self.F[i] += valuesInTriangle[6]
                    self.F[i] += valuesInTriangle[7]
                if (t == 5):  # 5 - right side
                    self.F[i] += valuesInTriangle[0]
                    self.F[i] += valuesInTriangle[1]
                    self.F[i] += valuesInTriangle[6]
                    self.F[i] += valuesInTriangle[7]
                if (t == 6):  # 6 - right bottom corner
                    self.F[i] += valuesInTriangle[0]
                    self.F[i] += valuesInTriangle[1]
                if (t == 7):  # 7 - bottom
                    self.F[i] += valuesInTriangle[0]
                    self.F[i] += valuesInTriangle[1]
                    self.F[i] += valuesInTriangle[2]
                    self.F[i] += valuesInTriangle[3]
                if (t == 8):  # 8 - normal middle
                    self.F[i] += valuesInTriangle[0]
                    self.F[i] += valuesInTriangle[1]
                    self.F[i] += valuesInTriangle[2]
                    self.F[i] += valuesInTriangle[3]
                    self.F[i] += valuesInTriangle[4]
                    self.F[i] += valuesInTriangle[5]
                    self.F[i] += valuesInTriangle[6]
                    self.F[i] += valuesInTriangle[7]

                self.F[i] = (float(self.s.TriangleArea) / 6) * self.F[i]

            else:  # cross point

                valuesInTriangle = np.zeros([4, 2])

                valuesInTriangle[0] += self.f0([x - halfShortTriangleSide, y - halfShortTriangleSide])
                valuesInTriangle[0] += self.f0([x - halfShortTriangleSide, y + halfShortTriangleSide])

                valuesInTriangle[1] += self.f0([x - halfShortTriangleSide, y + halfShortTriangleSide])
                valuesInTriangle[1] += self.f0([x + halfShortTriangleSide, y + halfShortTriangleSide])

                valuesInTriangle[2] += self.f0([x + halfShortTriangleSide, y + halfShortTriangleSide])
                valuesInTriangle[2] += self.f0([x + halfShortTriangleSide, y - halfShortTriangleSide])

                valuesInTriangle[3] += self.f0([x + halfShortTriangleSide, y - halfShortTriangleSide])
                valuesInTriangle[3] += self.f0([x - halfShortTriangleSide, y - halfShortTriangleSide])

                self.F[i] += valuesInTriangle[0]
                self.F[i] += valuesInTriangle[1]
                self.F[i] += valuesInTriangle[2]
                self.F[i] += valuesInTriangle[3]

                self.F[i] = (float(self.s.TriangleArea) / 6) * self.F[i]

        for i in range(0, self.s.indNumber()):
            for e in range(-self.s.BorderEdgesD - self.s.BorderEdgesN, -self.s.BorderEdgesD):
                e1 = int(self.s.Edges[e][0])
                e2 = int(self.s.Edges[e][1])
                p1 = self.s.Points[int(e1)][0:2]
                p2 = self.s.Points[int(e2)][0:2]
                if (i == e1 or i == e2):
                    self.F[i] += ((self.s.longTriangleSide * 0.5) * self.fN((p1 + p2) * 0.5))

        self.Zero = self.F[:, 0]
        self.One = self.F[:, 1]
