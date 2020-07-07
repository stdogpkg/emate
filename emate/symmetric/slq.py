"""
Sthocastic Lanczos Quadrature
=============================

Given a semi-positive definite matrix :math:`A \in \mathbb R^{|V|\\times|V|}`,
which has the set of eigenvalues given by :math:`\{\lambda_i\}` a trace of
a matrix function is given by

.. math:: 

    \mathrm{tr}(f(A)) = \sum\limits_{i=0}^{|V|} f(\lambda_i)

The methods for calculating such traces functions have a
cubic computational complexity lower bound,  :math:`O(|V|^3)`.
Therefore, it is not feasible for  large networks. One way
to overcome such computational complexity it is use stochastic approximations
combined with a mryiad of another methods
to get the results with enough accuracy and with a small computational cost. 
The methods available in this module uses the Sthocastic Lanczos Quadrature, 
a procedure proposed in the work made by Ubaru, S. et.al. [1] (you need to cite them).

References
----------

    [1] Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of tr(f(A)) via
    Stochastic Lanczos Quadrature. SIAM Journal on Matrix Analysis and
    Applications, 38(4), 1075-1099.

    [2] Hutchinson, M. F. (1990). A stochastic estimator of the trace of the
    influence matrix for laplacian smoothing splines. Communications in
    Statistics-Simulation and Computation, 19(2), 433-450.
"""
import numpy as np
import tensorflow as tf

from emate.utils.tfops.vector_factories import radamacher
from emate.symmetric.tfops.slq import trace_estimator


def pyslq(
    A,
    num_vecs,
    num_steps,
    trace_function,
    device='/gpu:0',
    precision=32,
    random_factory=radamacher,
    parallel_iterations=10,
    swap_memory=False,
    infer_shape=False,
):
    """
    Compute the approxiamted  value of a given trace function
    using the sthocastic Lanczos quadrature using Radamacher's random 
    vectors.

    Parameters
    ----------

        A : scipy sparse matrix
            The semi-positive definite matrix
        num_vecs: int
            Number of random vectors in oder to aproximate the 
            trace
        num_steps: int 
            Number of Lanczos steps
        trace_function: function
            A function like

            .. code-block:: python
            
                def trace_function(eig_vals)
                    *tensorflow ops
                    return result

        precision: int 
            (32) Single or (64) double precision

    Returns
    -------

        f_estimation: float 
            The approximated value of the given trace function 

        gammas: array of floats
            See [1] for more details
   
    References
    ----------

        [1] Ubaru, S., Chen, J., & Saad, Y. (2017).
        Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature. 
        SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.
    """
    coo = A.tocoo()

    if precision == 32:
        np_type = np.float32
        tf_type = tf.float32
    else:
        np_type = np.float64
        tf_type = tf.float64

    sp_values = np.array(coo.data, dtype=np_type)
    sp_indices = np.mat([coo.row, coo.col], dtype=np.int64).transpose()

    feed_dict = {
        "sp_values:0": sp_values,
        "sp_indices:0": sp_indices,
    }

    dimension = A.shape[0]

    tf.compat.v1.reset_default_graph()
    with tf.device(device):
        sp_indices = tf.compat.v1.placeholder(dtype=tf.int64, name="sp_indices")
        sp_values = tf.compat.v1.placeholder(
            dtype=tf_type,
            name="sp_values"
        )
        A = tf.SparseTensor(
            sp_indices,
            sp_values,
            dense_shape=np.array(A.shape, dtype=np.int32)
        )

        f_estimation, gammas = trace_estimator(
            A,
            dimension,
            num_vecs,
            num_steps,
            trace_function,
            random_factory,
            parallel_iterations,
            swap_memory,
            infer_shape
        )

    with tf.compat.v1.Session() as sess:
        f_estimation, gammas = sess.run([f_estimation, gammas], feed_dict)

    return f_estimation, gammas


__all__ = ["pyslq"]
