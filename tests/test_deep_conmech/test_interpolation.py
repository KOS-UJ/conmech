import numpy as np

from deep_conmech.data.interpolation_helpers import interpolate_scaled_nodes_numba


def test_interpolate_scaled_nodes_2d():
    scaled_nodes = np.random.uniform(size=(1000, 2))

    # vector = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    vector = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.float64)
    interpolated_scaled_nodes = interpolate_scaled_nodes_numba(
        scaled_nodes=scaled_nodes, corner_vectors=vector
    )
    assert np.allclose(scaled_nodes, interpolated_scaled_nodes)

    # assert np.allclose(np.min(interpolated_scaled_nodes, axis=0), np.min(vector, axis=0))
    # assert np.allclose(np.max(interpolated_scaled_nodes, axis=0), np.max(vector, axis=0))
    # assert np.allclose(np.mean(interpolated_scaled_nodes, axis=0), [0.5, 0.5], atol=0.1)


def test_interpolate_scaled_nodes_3d():
    scaled_nodes = np.random.uniform(size=(1000, 3))

    vector = np.array(
        [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
        dtype=np.float64,
    )
    interpolated_scaled_nodes = interpolate_scaled_nodes_numba(
        scaled_nodes=scaled_nodes, corner_vectors=vector
    )
    assert np.allclose(scaled_nodes, interpolated_scaled_nodes)
