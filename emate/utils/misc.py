import tensorflow as tf


def replace_by_indices(input_matrix, values, indices, dimension,
                       tf_float=tf.float32):
    with tf.name_scope("replace_by_indices"):

        identity_matrix = tf.eye(dimension, dtype=tf_float)

        mask_values = tf.tile([0.0], [tf.shape(indices)[0]])
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
            tf.shape(input_matrix, out_type=tf.int64)
        )
        output_matrix = tf.add(
            masked_input_tensor,
            tf.sparse_tensor_dense_matmul(
                values_sparse_tensor,
                identity_matrix
            )
        )

        return output_matrix
