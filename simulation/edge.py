"""
Created at 22.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""


class Edge:
    @staticmethod
    def c(edge):
        if edge[2] == 1:  # 1 - from normal go right to normal
            c1i = 3
            c1j = 0
            c2i = 4
            c2j = 7
        elif edge[2] == 2:  # 2 - from normal go up to normal
            c1i = 1
            c1j = 6
            c2i = 2
            c2j = 5
        elif edge[2] == 3:  # 3 - from normal go right and up to cross
            c1i = 2
            c1j = 0
            c2i = 3
            c2j = 3
        elif edge[2] == 4:  # 4 - from cross go right and up to normal
            c1i = 1
            c1j = 7
            c2i = 2
            c2j = 6
        elif edge[2] == 5:  # 5 - from normal go right and down to cross
            c1i = 4
            c1j = 1
            c2i = 5
            c2j = 0
        elif edge[2] == 6:  # 6 - from cross go right and down to normal
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
