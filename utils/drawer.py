"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab


class Drawer:

    def __init__(self, solv):
        self.solv = solv
        self.s = solv.s

    def draw(self):

        txt = 'CROSS EQUATION GR' + str(self.s.SizeH) + ' ' + str(self.s.SizeL) \
              + ' (ml ' + str(self.solv.mi) + " " + str(self.solv.la) \
              + ') F0[' + str(self.solv.F.F0[0]) + ',' + str(self.solv.F.F0[1]) \
              + '] FN[' + str(self.solv.F.FN[0]) + ',' + str(self.solv.F.FN[1]) + ']'

        plt.close()
        pylab.axes().set_aspect('equal', 'box')

        shadow = 0.1
        thickness1 = thickness2 = 2

        i = len(self.s.Edges) - 1
        j = len(self.s.Edges) - self.s.BorderEdgesD - 1
        while (j < i):
            x1 = self.s.Points[int(self.s.Edges[i, 0])][0]
            y1 = self.s.Points[int(self.s.Edges[i, 0])][1]
            x2 = self.s.Points[int(self.s.Edges[i, 1])][0]
            y2 = self.s.Points[int(self.s.Edges[i, 1])][1]
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=shadow, lw=thickness1)
            i -= 1
        j -= self.s.BorderEdgesN
        while (j < i):
            x1 = self.s.Points[int(self.s.Edges[i, 0])][0]
            y1 = self.s.Points[int(self.s.Edges[i, 0])][1]
            x2 = self.s.Points[int(self.s.Edges[i, 1])][0]
            y2 = self.s.Points[int(self.s.Edges[i, 1])][1]
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=shadow, lw=thickness1)
            i -= 1
        j -= self.s.BorderEdgesC
        while (j < i):
            x1 = self.s.Points[int(self.s.Edges[i, 0])][0]
            y1 = self.s.Points[int(self.s.Edges[i, 0])][1]
            x2 = self.s.Points[int(self.s.Edges[i, 1])][0]
            y2 = self.s.Points[int(self.s.Edges[i, 1])][1]
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=shadow, lw=thickness1)
            i -= 1
        while (-1 < i):
            x1 = self.s.Points[int(self.s.Edges[i, 0])][0]
            y1 = self.s.Points[int(self.s.Edges[i, 0])][1]
            x2 = self.s.Points[int(self.s.Edges[i, 1])][0]
            y2 = self.s.Points[int(self.s.Edges[i, 1])][1]
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=shadow, lw=thickness1)
            i -= 1

            # ------------

        # plt.scatter(self.solver.DisplacedPoints[:,0],self.solver.DisplacedPoints[:,1], marker='o')

        i = len(self.s.Edges) - 1
        j = len(self.s.Edges) - self.s.BorderEdgesD - 1
        while (j < i):
            x1 = self.solv.DisplacedPoints[int(self.s.Edges[i, 0])][0]
            y1 = self.solv.DisplacedPoints[int(self.s.Edges[i, 0])][1]
            x2 = self.solv.DisplacedPoints[int(self.s.Edges[i, 1])][0]
            y2 = self.solv.DisplacedPoints[int(self.s.Edges[i, 1])][1]
            plt.plot([x1, x2], [y1, y2], 'r-', lw=thickness2)
            i -= 1
        j -= self.s.BorderEdgesN
        while (j < i):
            x1 = self.solv.DisplacedPoints[int(self.s.Edges[i, 0])][0]
            y1 = self.solv.DisplacedPoints[int(self.s.Edges[i, 0])][1]
            x2 = self.solv.DisplacedPoints[int(self.s.Edges[i, 1])][0]
            y2 = self.solv.DisplacedPoints[int(self.s.Edges[i, 1])][1]
            plt.plot([x1, x2], [y1, y2], 'b-', lw=thickness2)
            i -= 1
        j -= self.s.BorderEdgesC
        while (j < i):
            x1 = self.solv.DisplacedPoints[int(self.s.Edges[i, 0])][0]
            y1 = self.solv.DisplacedPoints[int(self.s.Edges[i, 0])][1]
            x2 = self.solv.DisplacedPoints[int(self.s.Edges[i, 1])][0]
            y2 = self.solv.DisplacedPoints[int(self.s.Edges[i, 1])][1]
            plt.plot([x1, x2], [y1, y2], 'y-', lw=thickness2)
            i -= 1
        while (-1 < i):
            x1 = self.solv.DisplacedPoints[int(self.s.Edges[i, 0])][0]
            y1 = self.solv.DisplacedPoints[int(self.s.Edges[i, 0])][1]
            x2 = self.solv.DisplacedPoints[int(self.s.Edges[i, 1])][0]
            y2 = self.solv.DisplacedPoints[int(self.s.Edges[i, 1])][1]
            plt.plot([x1, x2], [y1, y2], 'k-', lw=thickness2)
            i -= 1

            # ------------

        # plt.savefig(txt + '.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)  # DPI 500
        # print(txt + '.png')
        plt.show()
        plt.close()
