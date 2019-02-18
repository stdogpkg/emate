"""
Lanczos
=======


Available methods
-----------------

trace_estimator


"""

import tensorflow as tf

from emate.utils.tfops.vector_factories import radamacher
from emate.utils.tfops.misc import scipy2tensor
from .tfops.slq import trace_estimator


def pytrace_estimator(
    A,
    dimension,
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

    tf.reset_default_graph()
    with tf.device(device):
        dimension = A.shape[0]
        A = scipy2tensor(A, precision=precision)

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
        f_estimation, gammas = sess.run([f_estimation, gammas])

    return f_estimation, gammas


__all__ = ["pytrace_estimator"]
