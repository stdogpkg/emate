"""
Lanczos
=======


Available methods
-----------------

trace_estimator


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
    random_factory=radamacher,
    precision=32,
    parallel_iterations=10,
    swap_memory=False,
    infer_shape=False,
    device='/gpu:0',
):
    """Sthocastic Lanczos Quadrature
    
    .. math::

      \mathrm{tr}\exp(L) = \sum\limits_{i=0}^{|V|} e^{\lambda_i}

    Parameters
    ----------
        L: sparse matrix
        num_vecs: int
            Number of  random vectors used to approximate the trace
            using the Hutchison's trick [1]
        num_steps: int
            Number of Lanczos steps or Chebyschev's moments
        device: str
            "/cpu:int" our "/gpu:int"
    
    Returns
    -------
        approximated_estrada_index: float

    References
    ----------

    .. [1]Ubaru, S., Chen, J., & Saad, Y. (2017).
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

    tf.reset_default_graph()
    with tf.device(device):
        sp_indices = tf.placeholder(dtype=tf.int64, name="sp_indices")
        sp_values = tf.placeholder(
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

    with tf.Session() as sess:
        f_estimation, gammas = sess.run([f_estimation, gammas], feed_dict)

    return f_estimation, gammas


__all__ = ["pyslq"]
