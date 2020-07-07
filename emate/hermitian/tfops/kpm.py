"""
Kernel Polynomial Method
========================

The kernel polynomial method is an algorithm to obtain an approximation
for the spectral density of a Hermitian matrix. This algorithm combines
expansion in polynomials of Chebyshev with the stochastic trace in order
to obtain such approximation.

Applications
------------

    - Hamiltonian matrices associated with quantum mechanics
    - Magnetic Laplacian associated with directed graphs
    - etc

Available functions
-------------------


"""
import numpy as np
import tensorflow as tf

from emate.utils.tfops.misc import sparse_tensor_dense_matmul_gpu


def get_moments(
    H,
    num_vecs,
    num_moments,
    alpha0,
    drop_moments_history=True,
    swap_memory=True,
    force_gpu=True,
    name_scope=None
):
    """
    Parameters
    ----------
        H: SparseTensor of rank 2
        num_vecs: (uint) number of random vectors
        num_moments: (uint) number of cheby. moments
        alpha0: Tensor(shape=(H.shape[0], num_vecs), dtype=tf_complex)
        alpha1: Tensor(shape=(H.shape[0], num_vecs), dtype=tf_complex)
        drop_moments_history: (bool) preserve or not the info. about the random
            vectors
        swap_memory: (bool) swap memory CPU-GPU

    Returns
    -------
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
    complex_eval = (H.dtype.is_complex) or (alpha0.dtype.is_complex)
    with tf.name_scope(name_scope, "get_moments"):
        i_moment = tf.constant(0)
        num_iter_while = tf.constant(int(num_moments//2-1))
        alpha1 = sparse_tensor_dense_matmul_gpu(H, alpha0, force_gpu=force_gpu)

        # first_moment.shape = (num_vecs, )
        if complex_eval:
            alpha0conj = tf.math.conj(alpha0)
        else:
            alpha0conj = alpha0

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

            matrix_mul = 2.*sparse_tensor_dense_matmul_gpu(H, alpha1,
                force_gpu=force_gpu)

            alpha2 = matrix_mul-alpha0

            if complex_eval:
                alpha1conj = tf.math.conj(alpha1)
                alpha2conj = tf.math.conj(alpha2)

            else:
                alpha1conj = alpha1
                alpha2conj = alpha2

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

        if drop_moments_history:
            return moments

        return moments, first_moment, second_moment, alpha1, alpha2


def apply_kernel(
    moments,
    kernel,
    dimension,
    num_moments,
    num_vecs,
    extra_points=1,
    drop_moments_history=True,
    name_scope=None
):
    """
    Parameters
    ----------

        tf_float: (tensorflow float type)
            valids values are tf.float32, tf.float64, or tf.float128
        name_scope: (str) (default="jackson_kernel")
            scope name for tensorflow


    """

    with tf.name_scope(name_scope, "apply_kernel"):

        moments = tf.math.real(moments)

        if drop_moments_history is False:
            moments = tf.reduce_sum(moments, axis=0)
            moments = tf.reshape(moments, (1, num_moments))

        moments = tf.reduce_sum(moments, axis=0)

        moments = moments/num_vecs/dimension
        moments = tf.reshape(moments, shape=(1, num_moments))

        if kernel is not None:
            moments = moments*kernel

        smooth_moments = tf.concat(
            [
                moments,
                tf.zeros((1, extra_points),
                         dtype=moments.dtype)
            ],
            axis=1
        )

        num_points = num_moments+extra_points

        smooth_moments = tf.reshape(smooth_moments, [num_points])
        smooth_moments = tf.signal.dct(smooth_moments, type=3)

        points = tf.range(num_points, dtype=tf.float32)

        ek = tf.cos(np.pi*(points+0.5)/(num_points))
        gk = np.pi*tf.sqrt(1.-ek**2)
        rho = tf.math.divide(smooth_moments, gk)

    return ek, rho


__all__ = [ "apply_kernel", "get_moments"]
