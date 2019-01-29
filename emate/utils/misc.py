
import tensorflow as tf
import numpy as np


def random_vec_factory(
    H,
    dimension,
    num_vecs,
    tf_float=tf.float32,
    tf_complex=tf.complex64,
    name_scope=None
):
    """
    Generates a set of complex random vectors
    Args:
        H: SparseTensor(shape=(dimension, dimension), dtype=tf_complex)
        dimension: (int) dimension matrix
        num_vecs: (int) number of random vectors
        tf_float: (tensorflow float type)
        tf_complex: (tensorflow complex type)
        name_scope: (str)(default="random_vec_factory")
            scope name for tensorflow

    Return:
        alpha0: Tensor(shape=(dimension, num_vecs), dtype=tf_complex)
        alpha1: Tensor(shape=(dimension, num_vecs), dtype=tf_complex)

    """
    with tf.name_scope(name_scope, "random_vec_factory") as scope:

        random_phases = 2.*np.pi*tf.random.uniform(
            [dimension, num_vecs],
            dtype=tf_float
        )
        alpha0_sin = tf.sin(
            random_phases
        )
        alpha0_cos = tf.cos(
            random_phases
        )
        alpha0 = tf.add(
            tf.cast(alpha0_cos, dtype=tf_complex),
            1j*tf.cast(alpha0_sin, dtype=tf_complex)
        )

        alpha1_sin = tf.sparse_tensor_dense_matmul(H, alpha0_sin)
        alpha1_cos = tf.sparse_tensor_dense_matmul(H, alpha0_cos)
        alpha1 = tf.add(
            tf.cast(alpha1_cos, dtype=tf_complex),
            1j*tf.cast(alpha1_sin, dtype=tf_complex)
        )
        return alpha0, alpha1
