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
try:
    import cupy as cp
except:
    cp = None

from emate.utils.cupyops.signal import dctIII


def get_moments(
    H_rescaled,
    num_moments,
    dimension,
    precision=32
):
    """
    Parameters
    ----------
        H: sparse cupy of rank 2
        num_moments: (uint) number of cheby. moments
        dimension: (uint) size of the matrix

    Returns
    -------
    """
    cp_complex = cp.complex64
    cp_float = "float32"
    if precision == 64:
        cp_complex = cp.complex128
        cp_float = "float64"
    print("cp complex", cp_complex)
    alpha0 = cp.exp(1j*2*cp.pi*cp.random.rand(dimension,dtype=cp_complex))
    alpha1 = H_rescaled.dot(alpha0)
    mu = cp.zeros(num_moments, dtype=cp_complex)
    mu[0] = (alpha0.T.conj()).dot(alpha0)
    mu[1] = (alpha0.T.conj()).dot(alpha1)

    for i_moment in range(1, num_moments//2):
        alpha2 = 2*H_rescaled.dot(alpha1)-alpha0
        mu[2*i_moment] = 2*(alpha1.T.conj()).dot(alpha1) - mu[0]
        mu[2*i_moment+1] = 2*(alpha2.T.conj()).dot(alpha1) - mu[1]

        alpha0 = alpha1
        alpha1 = alpha2

    return mu


def apply_kernel(
    moments,
    kernel,
    num_moments,
    num_vecs,
    extra_points=1,
    precision=32
):
    """
    Parameters
    ----------
    Apply a given kernel in a given array of moments.
    Return the cosine transform of type III.
    """

    num_points = extra_points+num_moments

    if kernel is not None:
        moments = moments*kernel

    mu_ext = cp.zeros(num_points, dtype=moments.dtype)
    mu_ext[0:num_moments] = moments

    smooth_moments = dctIII(mu_ext, precision)
    points = cp.arange(0, num_points)
    ek = cp.cos(cp.pi*(points+0.5)/num_points)
    gk = cp.pi*cp.sqrt(1.-ek**2)

    rho = cp.divide(smooth_moments, gk)

    return ek, rho


__all__ = ["apply_kernel", "get_moments"]
