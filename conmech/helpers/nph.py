# NumPy Helpers
import numba
DIM = 2

def stack(data):
    return data.T.flatten()

def stack_column(data):
    return data.T.flatten().reshape(-1, 1)

def unstack(data):
    return data.reshape(-1, DIM, order="F")


@numba.njit(inline='always')
def div_or_zero_numba(value, denominator):
    return value / denominator if denominator != 0 else 0.0
