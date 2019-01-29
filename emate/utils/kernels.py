import tensorflow as tf
import numpy as np

def get_jackson_kernel(
        num_moments,
        tf_float,
        name_scope=None):
    """
    This function generates the Jackson kernel for a given  number of
    Chebyscev moments

    Args:
        num_moments: (uint) number of Chebyshev moments
        tf_float: (tensorflow float type) valids values are tf.float32
            tf.float64, or tf.float128
        name_scope: (str) (default="get_jackson_kernel")
            scope name for tensorflow

    Return:
        jackson_kernel: Tensor(shape=(num_moments,), dtype=tf_float)

    Note:
        See .. _The Kernel Polynomial Method:
        https://arxiv.org/pdf/cond-mat/0504627.pdf for more details
    """
    with tf.name_scope(name_scope, "get_jackson_kernel") as scope:

        kernel_moments = tf.range(0, num_moments, dtype=tf_float)
        phases = 2.*np.pi*kernel_moments/(num_moments+1)

        jackson_kernel = tf.div(
                tf.add(
                    (num_moments-kernel_moments+1)*tf.cos(phases),
                    tf.sin(phases)/tf.tan(np.pi/(num_moments+1))
                ),
                (num_moments+1)
            )
        return jackson_kernel
