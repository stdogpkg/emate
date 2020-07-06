"""
Utils TF
================


Available methods
-----------------

    - replace_by_indices
        Given an input_matrix, replaces the values given a set of indices.

"""
import tensorflow as tf
import numpy as np


def break_sparse_tensor(a):

    imag_values = tf.math.imag(a.values)
    real_values = tf.math.real(a.values)
    imag = tf.SparseTensor(a.indices, imag_values, a.dense_shape)
    real = tf.SparseTensor(a.indices, real_values, a.dense_shape)

    return real, imag


def sparse_tensor_dense_matmul_gpu(
    sp_a,
    b,
    force_gpu=False,
    adjoint_a=False,
    adjoint_b=False
):

    sp_a_is_complex = sp_a.dtype.is_complex
    b_is_complex = b.dtype.is_complex
    ab_are_real = sp_a_is_complex is False and b_is_complex is False

    if ab_are_real or force_gpu is False:
        result = tf.sparse.sparse_dense_matmul(sp_a, b, adjoint_a, adjoint_b)

    elif sp_a_is_complex is False and b_is_complex:
        real_b = tf.math.real(b)
        imag_b = tf.math.imag(b)

        imag = tf.sparse.sparse_dense_matmul(sp_a, imag_b, adjoint_a,
            adjoint_b)
        real = tf.sparse.sparse_dense_matmul(sp_a, real_b, adjoint_a,
            adjoint_b)

        result = tf.add(
            tf.cast(real, dtype=b.dtype),
            1j*tf.cast(imag, dtype=b.dtype)
        )

    elif b_is_complex is False and sp_a_is_complex:
        real_a, imag_a = break_sparse_tensor(sp_a)

        imag = tf.sparse.sparse_dense_matmul(imag_a, b, adjoint_a,
            adjoint_b)
        real = tf.sparse.sparse_dense_matmul(real_a, b, adjoint_a,
            adjoint_b)

        result = tf.add(
            tf.cast(real, dtype=sp_a.dtype),
            1j*tf.cast(imag, dtype=sp_a.dtype)
        )

    elif b_is_complex and sp_a_is_complex:
        real_a, imag_a = break_sparse_tensor(sp_a)

        real_b = tf.math.real(b)
        imag_b = tf.math.imag(b)

        #pure imaginary

        imag_a_real_b = tf.sparse.sparse_dense_matmul(imag_a, real_b, adjoint_a,
            adjoint_b)
        real_a_imag_b = tf.sparse.sparse_dense_matmul(real_a, imag_b, adjoint_a,
            adjoint_b)
        imag = tf.add(imag_a_real_b, real_a_imag_b)

        # real
        real_a_real_b = tf.sparse.sparse_dense_matmul(real_a, real_b, adjoint_a,
            adjoint_b)

        imag_a_imag_b = tf.sparse.sparse_dense_matmul(imag_a, imag_b, adjoint_a,
            adjoint_b)
        real = tf.subtract(real_a_real_b, imag_a_imag_b )

        result = tf.add(
            tf.cast(real, dtype=sp_a.dtype),
            1j*tf.cast(imag, dtype=sp_a.dtype)
        )

    return result


def replace_by_indices(input_matrix, values, indices, name_scope=None):
    """ Given an input_matrix, replaces the values given a set of indices.

    Parameters
    ----------
        input_matrix: Tensor(shape=(dimension, dimension), dtype=tf_type)
        values: Tensor(shape=(None,), dtype=tf_type)
        indices: Tensor(shape=(None, 2), dtype=int64)
        name_scope: (str)(default="replace_by_indices")
            scope name for tensorflow

    Return
    ------
        output_matrix: Tensor(shape=(dimension, dimension), dtype=tf_type)

    """
    tf_type = input_matrix.dtype
    with tf.name_scope("replace_by_indices", name_scope):

        identity_matrix = tf.eye(
            tf.shape(input_matrix, out_type=tf.int64)[1],
            dtype=tf_type
        )
        ones = tf.ones(tf.shape(input_matrix), dtype=tf_type)

        mask_sparse_tensor_ones = tf.SparseTensor(
            indices,
            values=tf.tile(
                tf.constant([-1.0], dtype=tf_type),
                [tf.shape(indices)[0]]
            ),
            dense_shape=tf.shape(input_matrix, out_type=tf.int64)
        )

        # tf.sparse.sparse_dense_matmul(.., I) can be replaced by
        # tf.sparse.to_dense. However, the last has no GPU suport until now
        mask_values = tf.add(
            ones,
            tf.sparse.sparse_dense_matmul(
                    mask_sparse_tensor_ones,
                    identity_matrix
                )
        )

        masked_input_tensor = tf.multiply(
            input_matrix,
            mask_values
        )

        values_sparse_tensor = tf.SparseTensor(
            indices,
            values,
            tf.shape(input_matrix, out_type=tf.int64),
        )
        output_matrix = tf.add(
            masked_input_tensor,
            tf.sparse.sparse_dense_matmul(
                values_sparse_tensor,
                identity_matrix
            )
        )

        return output_matrix


def scipy2tensor(scipy_sp_a, precision=32):
    coo = scipy_sp_a.tocoo()
    if np.isrealobj(coo.data):
        if precision == 32:
            np_type = np.float32
        elif precision == 64:
            np_type = np.float64
    else:
        if precision == 32:
            np_type = np.complex64
        elif precision == 64:
            np_type = np.complex128

    data = np.array(coo.data, dtype=np_type)

    shape = np.array(coo.shape, dtype=np.int32)
    indices = np.mat([coo.row, coo.col], dtype=np.float32).transpose()
    sp_a = tf.SparseTensor(indices, data, shape)
    return sp_a


__all__ = [
    "break_sparse_tensor",
    "sparse_tensor_dense_matmul_gpu",
    "replace_by_indices",
    "scipy2tensor"
]
