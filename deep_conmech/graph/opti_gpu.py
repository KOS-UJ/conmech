import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

minimum = np.array([1.0, 1.0])
scales = np.array([2.0, 3.0])


def loss_and_gradient(x):
    return tfp.math.value_and_gradient(
        lambda x: tf.reduce_sum(
            scales * tf.math.squared_difference(x, minimum), axis=-1),
        x)


start = tf.constant([0.6, 0.8])  # Starting point for the search.
optim_results = tfp.optimizer.bfgs_minimize(
    loss_and_gradient, initial_position=start, tolerance=1e-8)

# Check that the search converged
assert (optim_results.converged)
# Check that the argmin is close to the actual value.
np.testing.assert_allclose(optim_results.position, minimum)
# Print out the total number of function evaluations it took. Should be 5.
print("Function evaluations: %d" % optim_results.num_objective_evaluations)
