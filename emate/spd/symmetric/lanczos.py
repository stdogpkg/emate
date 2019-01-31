import tensorflow as tf
from emate.utils.vector_factories import radamacher
from emate.linalg.lanczos import lanczos
import numpy as np

def trace_estimator(
    A,
    dimension,
    num_vecs,
    num_steps,
    trace_function,
    random_factory=radamacher,
    parallel_iterations=10,
    swap_memory=False,
    infer_shape=False,
    tf_float=tf.float32,
):
    with tf.name_scope("Trace_Estimator") as scope:
        with tf.name_scope("init_vars") as scope:
            nv = tf.constant(num_vecs, dtype=tf_float, name="nv")
            n = tf.constant(dimension, dtype=tf_float, name="n")
            factor = tf.divide(n, nv, "n/nv")

        with tf.name_scope("sthocastic_step") as scope:
            def sthocastic_step(i_step):
                with tf.name_scope("init_vars") as scope:
                    v0 = random_factory((dimension, 1), tf_float)

                V, alphas, betas, ortho_ok = lanczos(A, dimension, v0, num_steps)

                with tf.name_scope("construct_tridiagonal") as scope:
                    with tf.name_scope("build_diag_matrices") as scope:
                        T = tf.diag(alphas)
                        B = tf.diag(betas)

                    with tf.name_scope("padd_and_roll_values") as scope:
                        paddings = tf.constant([[0, 1,], [0, 1]])
                        B =  tf.pad(B, paddings, "CONSTANT")
                        above = tf.roll(B, shift=1, axis=0)
                        below = tf.roll(B, shift=1, axis=1)

                    with tf.name_scope("create_tridiagonal") as scope:
                        T = T +  above + below

                    eigvals, eigvecs = tf.linalg.eigh(
                        T,
                    )
                    thetas = trace_function(eigvals)
                    tau2 = tf.math.pow(tf.gather(eigvecs, 0), 2)

                    gamma = tf.reduce_sum(
                        tf.math.multiply(
                            tau2,
                            thetas
                        ),
                        axis=0
                    )
                return gamma

        gammas = tf.map_fn(
            sthocastic_step,
            np.arange(0, num_vecs),
            dtype=(tf_float),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            infer_shape=infer_shape,
            name="map_gammas_sthocastic_step"
        )

        f_estimation = tf.reduce_sum(
            factor*gammas, axis=0 , name="f_estimation"
        )
        return f_estimation, gammas
