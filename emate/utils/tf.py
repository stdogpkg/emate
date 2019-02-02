"""
Utils TF
================


Available methods
-----------------

    - replace_by_indices
        Given an input_matrix, replaces the values given a set of indices.

"""


import tensorflow as tf


def replace_by_indices(input_matrix, values, indices,
                       tf_type=tf.float32, name_scope=None):
    """
    Given an input_matrix, replaces the values given a set of indices.

    Parameters
    ----------
        input_matrix: Tensor(shape=(dimension, dimension), dtype=tf_type)
        values: Tensor(shape=(None,), dtype=tf_type)
        indices: Tensor(shape=(None, 2), dtype=int64)
        tf_type: (tensorflow type)(default=tf.float32)
        name_scope: (str)(default="replace_by_indices")
            scope name for tensorflow

    Return
    ------
        output_matrix: Tensor(shape=(dimension, dimension), dtype=tf_type)

    """

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

        # tf.sparse_tensor_dense_matmul(.., I) can be replaced by
        # tf.sparse.to_dense. However, the last has no GPU suport until now
        mask_values = tf.add(
            ones,
            tf.sparse_tensor_dense_matmul(
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
            tf.sparse_tensor_dense_matmul(
                values_sparse_tensor,
                identity_matrix
            )
        )

        return output_matrix
