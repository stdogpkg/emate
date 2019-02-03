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
    return_complex=False,
    tf_float=tf.float32,
    tf_complex=tf.complex64,
    name_scope=None
):
    r"""Generates a set of complex random vectors
    .. math::
        v = [e^{2\pi i \phi_0}\dots e^{2\pi i \phi_n}]

    Parameters
    ----------
        shape: (int, int) dimension matrix
        return_complex: (boll)(default=False)
            The vast majority of methods implemented in tf only works with
            float types. Therefore, sometimes it most convinient perform all
            calculations using imag and real part of vectors, and after that
            peform the  operation.
        tf_float: (tensorflow float type)
        tf_complex: (tensorflow complex type)
        name_scope: (str)(default="random_vec_factory")
            scope name for tensorflow

    Returns
    ------
        alpha0: Tensor(shape=(dimension, num_vecs), dtype=tf_complex)
        alpha1: Tensor(shape=(dimension, num_vecs), dtype=tf_complex)

    """
    if return_complex:
        tf_type = tf_complex
    else:
        tf_type = tf_float

    with tf.name_scope(name_scope, "random_vec_factory"):

        random_phases = 2.*np.pi*tf.random.uniform(
            shape,
            dtype=tf_type
        )
        if return_complex:
            return tf.exp(1j*random_phases)

        else:
            vec_sin = tf.sin(
                random_phases
            )
            vec_cos = tf.cos(
                random_phases
            )
            return vec_cos, vec_sin


def radamacher(shape, norm=True, tf_float=tf.float32, name_scope=None):
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
    with tf.name_scope("radamacher_factory", name_scope):
        vec = tf.sign(tf.random.normal(shape, dtype=tf_float))
        if norm:
            vec = tf.divide(vec, tf.norm(vec))
        return vec
