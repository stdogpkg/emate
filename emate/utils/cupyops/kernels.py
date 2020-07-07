"""
Kernel Functions
================

Theses kernels functions are most used for Kernel Polynomial Method
in order to....


Available methods
-----------------

    - jackson
"""

try:
    import cupy as cp
except:
    cp = None


def jackson(
    num_moments,
    precision=32,
):
    """
    This function generates the Jackson kernel for a given  number of
    Chebyscev moments

    Parameters
    ----------
        num_moments: (uint)
            number of Chebyshev moments
    Return
    ------
        jackson_kernel: cupy array(shape=(num_moments,), dtype=tf_float)

    Note
    ----
        See .. _The Kernel Polynomial Method:
        https://arxiv.org/pdf/cond-mat/0504627.pdf for more details
    """
    cp_float = cp.float64
    if precision == 32:
        cp_float = cp.float32


    kernel_moments = cp.arange(num_moments, dtype=cp_float)
    norm = cp.pi/(num_moments+1)
    phase_vec = norm*kernel_moments
    kernel = (num_moments-kernel_moments+1)*cp.cos(phase_vec)
    kernel = kernel + cp.sin(phase_vec)/cp.tan(norm)
    kernel = kernel/(num_moments + 1)

    return kernel


__all__ = ["jackson"]
