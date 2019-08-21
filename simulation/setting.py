"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np

class Setting:

    def __init__(self):
        self.Points = np.zeros([0, 3])
        #
        # x, y, type:
        # 0 - left bottom corner, 1 - left side, 2 - left top corner,
        # 3 - top, 4 - right top corner, 5 - right side, 6 - right bottom corner, 7 - bottom,
        # 8 - normal middle, 9 - cross
        #
        self.Edges = np.zeros([0, 3])
        #
        # i, j, type: (always i<j on plane)
        # 0 - no edge
        # 1 - from normal go right to normal, 2 - from normal go up to normal,
        # 3 - from normal go right and up to cross, 4 - from cross go right and up to normal,
        # 5 - from normal go right and down to cross, 6 - from cross go right and down to normal
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

    def indNumber(self):
        return len(self.Points) - self.BorderEdgesD - 1

    def addPoint(self, x, y, t):
        i = 0
        while (i < len(self.Points)):
            if (self.Points[i][0] == x and self.Points[i][1] == y):
                return
            else:
                i += 1
        self.Points = np.append([[x, y, t]], self.Points, axis=0)
        for i in range(0, len(self.Edges)):
            self.Edges[i][0] += 1
            self.Edges[i][1] += 1

    def getPoint(self, x, y):
        i = 0
        while (i < len(self.Points)):
            if (self.Points[i][0] == x and self.Points[i][1] == y):
                return i
            else:
                i += 1
        return -1

    def addEdge(self, i, j, t):  # zawsze(i,j) ma i<j na x lub x równe oraz i<j na y
        a = i
        b = j
        if (self.Points[j][0] < self.Points[i][0] or
                (self.Points[j][0] == self.Points[i][0] and self.Points[j][1] < self.Points[i][1])):
            a = j
            b = i
        self.Edges = np.append([[a, b, t]], self.Edges, axis=0)

    def getEdgeType(self, i, j):  # kolejność argumentów ma znaczenie
        for e in self.Edges:
            if (e[0] == i and e[1] == j):
                return e[2]
        return 0

    def startBorder(self, x, y):
        self.addPoint(x, y, 0)

    def addBorderD(self, x, y):
        self.addPoint(x, y, 1)
        self.addEdge(1, 0, 2)
        self.BorderEdgesD += 1

    def addBorderDLast(self, x, y):
        self.addPoint(x, y, 2)
        self.addEdge(1, 0, 2)
        self.BorderEdgesD += 1

    def addBorderNTop(self, x, y):
        self.addPoint(x, y, 3)
        self.addEdge(1, 0, 1)
        self.BorderEdgesN += 1

    def addBorderNTopLast(self, x, y):
        self.addPoint(x, y, 4)
        self.addEdge(1, 0, 1)
        self.BorderEdgesN += 1

    def addBorderNSide(self, x, y):
        self.addPoint(x, y, 5)
        self.addEdge(0, 1, 2)
        self.BorderEdgesN += 1

    def addBorderNSideLast(self, x, y):
        self.addPoint(x, y, 6)
        self.addEdge(0, 1, 2)
        self.BorderEdgesN += 1

    def addBorderC(self, x, y):
        self.addPoint(x, y, 7)
        self.addEdge(0, 1, 1)
        self.BorderEdgesC += 1

    def stopBorder(self):
        self.addEdge(len(self.Points) - 1, 0, 1)
        self.BorderEdgesC += 1

    def construct(self, sizeH, sizeL, height):
        self.SizeH = sizeH
        self.SizeL = sizeL
        self.Height = height
        self.longTriangleSide = float(height) / sizeH
        self.Length = self.longTriangleSide * sizeL

        self.halfLongTriangleSide = float(self.longTriangleSide) * 0.5
        self.shortTriangleSide = float(self.longTriangleSide) * np.sqrt(2) * 0.5
        self.halfShortTriangleSide = float(self.shortTriangleSide) * 0.5
        self.TriangleArea = (self.longTriangleSide * self.longTriangleSide) / 4.

        self.startBorder(0, 0)

        for i in range(1, sizeH):
            self.addBorderD(0, float(i) * self.longTriangleSide)
        self.addBorderDLast(0, float(sizeH) * self.longTriangleSide)

        for i in range(1, sizeL):
            self.addBorderNTop(float(i) * self.longTriangleSide, height)
        self.addBorderNTopLast(float(sizeL) * self.longTriangleSide, height)

        for i in range(sizeH - 1, 0, -1):
            self.addBorderNSide(self.Length, float(i) * self.longTriangleSide)
        self.addBorderNSideLast(self.Length, float(0))

        for i in range(sizeL - 1, 0, -1):
            self.addBorderC(float(i) * self.longTriangleSide, 0)

        self.stopBorder()

        for i in range(0, sizeL):
            for j in range(1, sizeH):
                x1 = float(i) * self.longTriangleSide
                x2 = float(i + 1) * float(self.longTriangleSide)
                y = float(j) * self.longTriangleSide
                self.addPoint(x1, y, 8)
                self.addPoint(x2, y, 8)
                a = self.getPoint(x1, y)
                b = self.getPoint(x2, y)
                self.addEdge(a, b, 1)

        for i in range(1, sizeL):
            for j in range(0, sizeH):
                x = float(i) * self.longTriangleSide
                y1 = float(j) * self.longTriangleSide
                y2 = float(j + 1) * self.longTriangleSide
                self.addPoint(x, y1, 8)
                self.addPoint(x, y2, 8)
                a = self.getPoint(x, y1)
                b = self.getPoint(x, y2)
                self.addEdge(a, b, 2)

        for i in range(0, sizeL):
            for j in range(0, sizeH):
                x = (float(i) + 0.5) * self.longTriangleSide
                y = (float(j) + 0.5) * self.longTriangleSide
                self.addPoint(x, y, 9)
                a = self.getPoint(x, y)
                b = self.getPoint((float(i)) * self.longTriangleSide, (float(j) + 1.0) * self.longTriangleSide)
                self.addEdge(a, b, 5)
                b = self.getPoint((float(i) + 1.0) * self.longTriangleSide, (float(j) + 1.0) * self.longTriangleSide)
                self.addEdge(a, b, 4)
                b = self.getPoint((float(i) + 1.0) * self.longTriangleSide, (float(j)) * self.longTriangleSide)
                self.addEdge(a, b, 6)
                b = self.getPoint((float(i)) * self.longTriangleSide, (float(j)) * self.longTriangleSide)
                self.addEdge(a, b, 3)