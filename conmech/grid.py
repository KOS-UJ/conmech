"""
Created at 22.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numba
import numpy as np


class Grid:
    # TODO: bad practice
    LEFT_BOTTOM_CORNER = 0
    LEFT_SIDE = 1
    LEFT_TOP_CORNER = 2
    TOP = 3
    RIGHT_TOP_CORNER = 4
    RIGHT_SIDE = 5
    RIGHT_BOTTOM_CORNER = 6
    BOTTOM = 7
    NORMAL_MIDDLE = 8
    CROSS = 9

    def __init__(self):
        self.Points = np.zeros([0, 3])
        self.Edges = np.zeros([0, 3], dtype=np.int32)
        self.edges_idx = None
        self.edges_start = None
        # TODO: Bad practice
        #  i, j, type: (always i<j on plane)
        #  0 - no edge
        #  1 - from normal go right to normal, 2 - from normal go up to normal,
        #  3 - from normal go right and up to cross, 4 - from cross go right and up to normal,
        #  5 - from normal go right and down to cross, 6 - from cross go right and down to normal
        #
        self.BorderEdgesD = 0
        self.BorderEdgesN = 0
        self.BorderEdgesC = 0
        self.Height = 0
        self.Length = 0
        self.SizeH = 0
        self.SizeL = 0
        self.longTriangleSide = 0
        self.halfLongTriangleSide = 0
        self.shortTriangleSide = 0
        self.halfShortTriangleSide = 0
        self.TriangleArea = 0

    @property
    def independent_nodes_count(self):
        return len(self.Points) - self.BorderEdgesD - 1

    @property
    def contact_num(self):
        return self.BorderEdgesC

    def getPoint(self, x, y):
        i = 0
        while i < len(self.Points):
            if self.Points[i][0] == x and self.Points[i][1] == y:
                return i
            else:
                i += 1
        return -1

    # TODO: order of args still matters
    def get_edge(self, i, j):
        return get_edge(self.Edges, self.edges_idx, self.edges_start, i, j)


@numba.njit()
def get_edge(edges, edges_idx, edges_start, i, j):
    result = np.full(3, -1, dtype=np.int32)

    for edge_i in range(edges_start[i], edges_start[i + 1]):
        if edges[edges_idx[edge_i], 1] == j:
            result = edges[edges_idx[edge_i], :]
            break
    return result
