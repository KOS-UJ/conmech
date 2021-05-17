"""
Created at 22.08.2019
"""


class Edge:
    START = 0
    STOP = 1
    TYPE = 2

    HORIZONTAL = 1
    VERTICAL = 2
    DIAGONAL_UP_TO_CROSS_POINT = 3
    DIAGONAL_DOWN_TO_CROSS_POINT = 5
    DIAGONAL_UP_TO_GRID_POINT = 4
    DIAGONAL_DOWN_TO_GRID_POINT = 6

    @staticmethod
    def c(edge):
        if edge[Edge.TYPE] == Edge.HORIZONTAL:
            c1i = 3
            c1j = 0
            c2i = 4
            c2j = 7
        elif edge[Edge.TYPE] == Edge.VERTICAL:
            c1i = 1
            c1j = 6
            c2i = 2
            c2j = 5
        elif edge[Edge.TYPE] == Edge.DIAGONAL_UP_TO_CROSS_POINT:
            c1i = 2
            c1j = 0
            c2i = 3
            c2j = 3
        elif edge[Edge.TYPE] == Edge.DIAGONAL_UP_TO_GRID_POINT:
            c1i = 1
            c1j = 7
            c2i = 2
            c2j = 6
        elif edge[Edge.TYPE] == Edge.DIAGONAL_DOWN_TO_CROSS_POINT:
            c1i = 4
            c1j = 1
            c2i = 5
            c2j = 0
        elif edge[Edge.TYPE] == Edge.DIAGONAL_DOWN_TO_GRID_POINT:
            c1i = 2
            c1j = 1
            c2i = 3
            c2j = 0
        else:
            c1i = -1
            c1j = -1
            c2i = -1
            c2j = -1

        result = (c1i, c1j, c2i, c2j)
        return result
