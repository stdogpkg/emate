"""
Vector Factories
================

Explicar o que e uma vector factory, onde e usado

Available methods
-----------------

    - normal_complex

    - radamacher
"""

import tensorflow as tf
import numpy as np


def normal_complex(
    shape,
    precision=32,
    name_scope=None
):
    r"""Generates a set of complex random vectors
    .. math::
        v = [e^{2\pi i \phi_0}\dots e^{2\pi i \phi_n}]

    Parameters
    ----------
        shape: (int, int) dimension matrix

        tf_complex: (tensorflow complex type)
        name_scope: (str)(default="random_vec_factory")
            scope name for tensorflow

    Returns
    ------
        vector: Tensor(shape=shape, dtype=tf_complex)

    """
    tf_float = tf.float64
    tf_complex = tf.complex128
    if precision == 32:
        tf_float = tf.float32
        tf_complex = tf.complex64
    with tf.name_scope(name_scope, "normal_complex"):

        random_phases = 2.*np.pi*tf.random.uniform(
            shape,
            dtype=tf_float
        )
        random_phases = tf.cast(random_phases, tf_complex)
        vector = tf.exp(1j*random_phases)
        return vector


def radamacher(shape, norm=True, precision=32, name_scope=None):
    """Generates a set of Radamacher vectors.

    Parameters
    ----------
        shape: SparseTensor(shape=shape, dtype=tf_float)
        norm: (bool)(default=True)
            If True the Radamacher vector returned is normalized
        tf_float: (tensorflow float type)(default=tf.float32)
        name_scope: (str)(default="random_vec_factory")
            scope name for tensorflow

    Return
    ------
        vec: Tensor(shape=shape, dtype=tf_float)

    """
    tf_float = tf.float64
    if precision == 32:
        tf_float = tf.float32

    with tf.name_scope("radamacher_factory", name_scope):
        vec = tf.sign(tf.random.normal(shape, dtype=tf_float))
        if norm:
            vec = tf.math.divide(vec, tf.norm(vec))
        return vec


__all__ = ["normal_complex", "radamacher"]
