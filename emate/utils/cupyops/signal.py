"""
Signal Processing Functions
================



Available methods
-----------------

    - dctIII
        Cosine transform of type III.
"""

try:
    import cupy as cp
except:
    cp = None


def dctIII(x, precision=32):
    """
    That implements the cosine transform of type III.
    Here, we are using the fact that  transformation it is just
     the inverse of the cosine trasnform of type II.

    Args:
    -----
        x: 1d cupy array
    Returns:
    --------
        x_transformed: 1d cupy array

    """

    dtype = "complex64"
    if precision == 64:
        dtype = "complex128"

    N = x.shape[0]
    c = cp.empty(N+1, dtype=dtype)

    phi = cp.exp(1j*cp.pi*cp.arange(N, dtype=dtype)/(2*N))
    c[:N] = phi*x
    c[N] = 0.0

    x_transformed = 2*N*cp.fft.irfft(c)[:N]

    return x_transformed
