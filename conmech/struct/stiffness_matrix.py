class StiffnessMatrix:
    DIMENSION = None

    def __init__(self, data):
        self.data = data

    def __iadd__(self, other):
        if isinstance(other, StiffnessMatrix):
            assert self.DIMENSION == other.DIMENSION
            self.data += other.data
        else:
            self.data += other

    def __add__(self, other):
        if isinstance(other, StiffnessMatrix):
            assert self.DIMENSION == other.DIMENSION
            return type(self)(self.data + other.data)
        return type(self)(self.data + other)

    def __radd__(self, other):
        return self + other

    def __imul__(self, other):
        self.data *= other

    def __mul__(self, other):
        return type(self)(self.data * other)

    def __rmul__(self, other):
        return self * other

    def copy(self):
        return type(self)(self.data.copy())

    def __getitem__(self, item):
        return self.data[item]

    def __matmul__(self, other):
        if isinstance(other, StiffnessMatrix):
            return self.data @ other.data
        return self.data @ other

    def __imatmul__(self, other):
        if isinstance(other, StiffnessMatrix):
            self.data @= other.data
        else:
            self.data @= other

    def __rmatmul__(self, other):
        return other @ self.data

    @property
    def T(self):
        return self.data.T


class SM1(StiffnessMatrix):
    DIMENSION = (1, 1)


class SM1to2(StiffnessMatrix):
    DIMENSION = (1, 2)


class SM2(StiffnessMatrix):
    DIMENSION = (2, 2)

    @property
    def SM1(self) -> SM1:
        x_len = self.data.shape[0] // 2
        y_len = self.data.shape[1] // 2
        return SM1(self.data[:x_len, :y_len])


class SM3(StiffnessMatrix):
    DIMENSION = (3, 3)
