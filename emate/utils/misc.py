import tensorflow as tf


def replace_by_indices(input_matrix, values, indices, dimension,
                       tf_type=tf.float32, name_scope=None):

    """
    Generates a set of Radamacher vectors
    Args:
        input_matrix: SparseTensor(shape=shape, dtype=tf_float)
        values:
        indices:
        dimension:
        tf_float: (tensorflow float type)(default=tf.float32)
        name_scope: (str)(default="random_vec_factory")
            scope name for tensorflow

    Return:
        vec: Tensor(shape=shape, dtype=tf_float)

    """
    with tf.name_scope("replace_by_indices", name_scope):

        identity_matrix = tf.eye(dimension, dtype=tf_type)

        mask_values = tf.tile(
            tf.constant([0.0], dtype=tf_type),
            [tf.shape(indices)[0]]
        )
        mask_sparse_tensor = tf.SparseTensor(
            indices,
            mask_values,
            tf.shape(input_matrix, out_type=tf.int64)
        )
        # tf.sparse_tensor_dense_matmul(.., I) can be replaced by
        # tf.sparse.todense. However, the last has no GPU suport until now
        masked_input_tensor = tf.multiply(
            input_matrix,
            tf.sparse_tensor_dense_matmul(mask_sparse_tensor, identity_matrix)
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
