"""
Kernel Functions
================

Theses kernels functions are most used for Kernel Polynomial Method
in order to....


Available methods
-----------------

    - jackson
    - lorentz
"""

import tensorflow as tf
import numpy as np


def jackson(
    num_moments,
    precision=32,
    name_scope=None
):
    """
    This function generates the Jackson kernel for a given  number of
    Chebyscev moments

    Parameters
    ----------
        num_moments: (uint)
            number of Chebyshev moments
        tf_float: (tensorflow float type)
            valids values are tf.float32, tf.float64, or tf.float128
        name_scope: (str) (default="get_jackson_kernel")
            scope name for tensorflow

    Return
    ------
        jackson_kernel: Tensor(shape=(num_moments,), dtype=tf_float)

    Note
    ----
        See .. _The Kernel Polynomial Method:
        https://arxiv.org/pdf/cond-mat/0504627.pdf for more details
    """
    tf_float = tf.float64
    if precision == 32:
        tf_float = tf.float32

    with tf.name_scope(name_scope, "jackson_kernel"):

        kernel_moments = tf.range(0, num_moments, dtype=tf_float)
        norm = np.pi/(num_moments+1)
        phases = kernel_moments*norm

        kernel = tf.math.divide(
                tf.add(
                    (num_moments-kernel_moments+1)*tf.cos(phases),
                    tf.sin(phases)/tf.tan(norm)
                ),
                (num_moments+1)
            )
        return kernel


def lorentz(
    num_moments,
    l,
    precision=32,
    name_scope=None
):
    """
    This function generates the Lorentz kernel for a given  number of
    Chebyscev moments and a positive real number, l

    Parameters
    ----------
        num_moments: (int)
            positive integer, number of Chebyshev moments
        l: (float)
            positve number,
        tf_float: (tensorflow float type)
            valids values are tf.float32, tf.float64, or tf.float128
        name_scope: (str) (default="lorentz_kernel")
            scope name for tensorflow

    Return
    ------
        kernel: Tensor(shape=(num_moments,), dtype=tf_float)

    Note
    ----
        See .. _The Kernel Polynomial Method:
        https://arxiv.org/pdf/cond-mat/0504627.pdf for more details
    """
    tf_float = tf.float64
    if precision == 32:
        tf_float = tf.float32

    with tf.name_scope(name_scope, "lorentz_kernel"):

        kernel_moments = tf.range(0, num_moments, dtype=tf_float)
        phases = 1. - kernel_moments/num_moments

        kernel = tf.math.divide(
                tf.sinh(l*phases),
                tf.math.sinh(l)
            )
        return kernel


__all__ = ["lorentz", "jackson"]
