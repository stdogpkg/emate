import tensorflow as tf
import numpy as np


def get_moments(
    H,
    num_vecs,
    num_moments,
    alpha0,
    alpha1,
    drop_moments_history=False,
    tf_float=tf.float32,
    tf_complex=tf.complex64,
    swap_memory=True,
    name_scope=None
):

    """
    Parameters:
        H: SparseTensor of rank 2
        num_vecs: (uint) number of random vectors
        num_moments: (uint) number of cheby. moments
        alpha0: Tensor(shape=(H.shape[0], num_vecs), dtype=tf_complex)
        alpha1: Tensor(shape=(H.shape[0], num_vecs), dtype=tf_complex)
        drop_moments_history: (bool) preserve or not the info. about the random
            vectors
        tf_float: tensorflow float dtype
        tf_complex: tensorflow complex dtyle
        swap_memory: (bool) swap memory CPU-GPU

    Return:
        moments: Tensor("while", shape=(?, num_moments), dtype=tf_complex)
            if drop_moments_history:
                ? = 1
            else:
                ? = num_vecs
        first_moment: Tensor("while", shape=(num_vecs,), dtype=tf_complex)
        second_moment: Tensor("while", shape=(num_vecs,), dtype=tf_complex)
        alpha1: Tensor("while", shape=(H.shape[0], num_vecs), dtype=tf_complex)
        alpha2: Tensor("while", shape=(H.shape[0], num_vecs), dtype=tf_complex)


    """
    with tf.name_scope(name_scope, "get_moments") as scope:
        i_moment = tf.constant(0)
        num_iter_while = tf.constant(int(num_moments//2-1))

        alpha0conj = tf.conj(alpha0)

        # first_moment.shape = (num_vecs, )
        first_moment = tf.reduce_sum(
                    tf.multiply(alpha0conj, alpha0),
                    axis=0
                )
        # second_moment.shape = (num_vecs, )
        second_moment = tf.reduce_sum(
            tf.multiply(alpha0conj, alpha1),
            axis=0
        )

        # moments.shape = (num_vecs, 2)
        moments = tf.concat(
            [
                tf.reshape(first_moment, (num_vecs, 1)),
                tf.reshape(second_moment, (num_vecs, 1))
            ],
            axis=1
        )
        if drop_moments_history:
            moments = tf.reduce_sum(moments, axis=0)

            # moments.shape = (num_vecs, 2)
            moments = tf.reshape(moments, (1, 2))
            while_shape_moments = tf.TensorShape([1, None])
        else:
            while_shape_moments = tf.TensorShape([num_vecs, None])

        def cond(
            moments,
            first_moment,
            second_moment,
            alpha0,
            alpha1,
            i_moment,
            num_iter_while
        ):
            return tf.less(i_moment, num_iter_while)

        def body(
            moments,
            first_moment,
            second_moment,
            alpha0,
            alpha1,
            i_moment,
            num_iter_while
        ):

            alpha1_imag = tf.math.imag(alpha1)
            alpha1_real = tf.math.real(alpha1)

            matrix_mul_real = 2*tf.sparse_tensor_dense_matmul(H, alpha1_real)
            matrix_mul_imag = 2*tf.sparse_tensor_dense_matmul(H, alpha1_imag)

            matrix_mul = tf.add(
                tf.cast(matrix_mul_real, tf_complex),
                1j*tf.cast(matrix_mul_imag, tf_complex),
            )
            alpha2 = matrix_mul-alpha0

            alpha2conj = tf.conj(alpha2)
            alpha1conj = tf.conj(alpha1)
            even = tf.reduce_sum(
                    tf.multiply(alpha1conj, alpha1),
                    axis=0
                )
            odd = tf.reduce_sum(
                tf.multiply(alpha2conj, alpha1),
                axis=0
            )
            even = 2.*even - first_moment
            odd = 2.*odd - second_moment

            event = tf.reshape(even, (num_vecs, 1))
            oddt = tf.reshape(odd, (num_vecs, 1))
            new_moments = tf.concat([event, oddt], axis=1)

            if drop_moments_history:
                new_moments = tf.reduce_sum(new_moments, axis=0)
                new_moments = tf.reshape(new_moments, (1, 2))

            return [
                tf.concat(
                    [moments, new_moments],
                    axis=1
                ),
                first_moment,
                second_moment,
                alpha1,
                alpha2,
                tf.add(i_moment, 1),
                num_iter_while
            ]

        (
            moments,
            first_moment,
            second_moment,
            alpha1,
            alpha2,
            i_moment,
            n_iter_moments
        ) = tf.while_loop(
            cond,
            body,
            [
                moments,
                first_moment,
                second_moment,
                alpha0,
                alpha1,
                i_moment,
                num_iter_while
            ],
            shape_invariants=[
                while_shape_moments,
                first_moment.get_shape(),
                second_moment.get_shape(),
                alpha0.get_shape(),
                alpha1.get_shape(),
                i_moment.get_shape(),
                num_iter_while.get_shape()
            ],
            swap_memory=swap_memory

        )

        return moments, first_moment, second_moment, alpha1, alpha2


def apply_kernel(
        moments,
        kernel,
        scale_fact_a,
        scale_fact_b,
        dimension,
        num_moments,
        num_vecs,
        extra_points=1,
        drop_moments_history=True,
        tf_float=tf.float32,
        name_scope=None
):

        with tf.name_scope(name_scope, "apply_kernel") as scope:

            moments = tf.math.real(moments)

            if drop_moments_history is False:
                moments = tf.reduce_sum(moments, axis=0)
                moments = tf.reshape(moments, (1, num_moments))

            moments = tf.reduce_sum(moments, axis=0)

            moments = moments/num_vecs/dimension
            moments = tf.reshape(moments, shape=(1, num_moments))

            smooth_moments = tf.concat(
                [
                    moments*kernel,
                    tf.zeros((1, extra_points),
                             dtype=tf_float)
                ],
                axis=1
            )

            num_points = num_moments+extra_points

            smooth_moments = tf.reshape(smooth_moments, [num_points])
            smooth_moments = tf.spectral.dct(smooth_moments, type=3)

            points = tf.range(num_points, dtype=tf_float)

            ek_rescaled = tf.cos(np.pi*(points+0.5)/(num_points))
            gk = np.pi*tf.sqrt(1.-ek_rescaled**2)

            ek = ek_rescaled*scale_fact_a + scale_fact_b
            rho = tf.divide(smooth_moments, gk)/scale_fact_a

        return ek, rho
