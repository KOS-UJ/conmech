# NumPy Helpers
DIM = 2

def stack(data):
    return data.T.flatten()

def stack_column(data):
    return data.T.flatten().reshape(-1, 1)

def unstack(data):
    return data.reshape(-1, DIM, order="F")
