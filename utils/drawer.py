"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import matplotlib.pyplot as plt
import pylab


class Drawer:

    def __init__(self, solv):
        self.solv = solv
        self.grid = solv.grid

    def draw(self):

        txt = 'CROSS EQUATION GR' + str(self.grid.SizeH) + ' ' + str(self.grid.SizeL) \
              + ' (ml ' + str(self.solv.mi) + " " + str(self.solv.la) \
              + ') F0[' + str(self.solv.F.F0[0]) + ',' + str(self.solv.F.F0[1]) \
              + '] FN[' + str(self.solv.F.FN[0]) + ',' + str(self.solv.F.FN[1]) + ']'

        plt.close()
        pylab.axes().set_aspect('equal', 'box')

        shadow = 0.1
        thickness1 = thickness2 = 2

        i = len(self.grid.Edges) - 1
        j = len(self.grid.Edges) - self.grid.BorderEdgesD - 1
        while j < i:
            x1, x2, y1, y2 = self.get_coordinates(self.grid.Points, i)
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=shadow, lw=thickness1)
            i -= 1
        j -= self.grid.BorderEdgesN
        while j < i:
            x1, x2, y1, y2 = self.get_coordinates(self.grid.Points, i)
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=shadow, lw=thickness1)
            i -= 1
        while self.grid.BorderEdgesC - 1 < i:
            x1, x2, y1, y2 = self.get_coordinates(self.grid.Points, i)
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=shadow, lw=thickness1)
            i -= 1
        while -1 < i:
            x1, x2, y1, y2 = self.get_coordinates(self.grid.Points, i)
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=shadow, lw=thickness1)
            i -= 1

            # ------------

        # plt.scatter(self.solver.DisplacedPoints[:,0],self.solver.DisplacedPoints[:,1], marker='o')

        i = len(self.grid.Edges) - 1
        j = len(self.grid.Edges) - self.grid.BorderEdgesD - 1
        while j < i:
            x1, x2, y1, y2 = self.get_coordinates(self.solv.DisplacedPoints, i)
            plt.plot([x1, x2], [y1, y2], 'r-', lw=thickness2)
            i -= 1
        j -= self.grid.BorderEdgesN
        while j < i:
            x1, x2, y1, y2 = self.get_coordinates(self.solv.DisplacedPoints, i)
            plt.plot([x1, x2], [y1, y2], 'b-', lw=thickness2)
            i -= 1
        while self.grid.BorderEdgesC - 1 < i:
            x1, x2, y1, y2 = self.get_coordinates(self.solv.DisplacedPoints, i)
            plt.plot([x1, x2], [y1, y2], 'k-', lw=thickness2)
            i -= 1
        while -1 < i:
            x1, x2, y1, y2 = self.get_coordinates(self.solv.DisplacedPoints, i)
            plt.plot([x1, x2], [y1, y2], 'y-', lw=thickness2)
            i -= 1

            # ------------

        # plt.savefig(txt + '.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)  # DPI 500
        # print(txt + '.png')
        plt.show()
        plt.close()

    def get_coordinates(self, array, i):
        x1 = array[int(self.grid.Edges[i, 0])][0]
        y1 = array[int(self.grid.Edges[i, 0])][1]
        x2 = array[int(self.grid.Edges[i, 1])][0]
        y2 = array[int(self.grid.Edges[i, 1])][1]

        return x1, x2, y1, y2
