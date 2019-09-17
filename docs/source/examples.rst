========
Examples
========

Kernel Polynomial Method (Chebyshev Polynomial expansion) 
=========================================================


The Kernel Polynomial Method can estimate the spectral density of large
sparse Hermitan matrices with a low computational cost. This method
combines three key ingredients: the Chebyshev expansion + the stochastic
trace estimator + kernel smoothing.


.. code-block:: python

    import igraph as ig
    import numpy as np

    N = 3000
    G = ig.Graph.Erdos_Renyi(N, 3/N)

    W = np.array(G.get_adjacency().data, dtype=np.float64)
    vals = np.linalg.eigvalsh(W).real

.. code-block:: python

    from emate.hermitian import pykpm
    from stdog.utils.misc import ig2sparse 

    W = ig2sparse(G)

    num_moments = 300
    num_vecs = 200
    extra_points = 10
    ek, rho = pykpm(W, num_moments, num_vecs, extra_points)

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.hist(vals, density=True, bins=100, alpha=.9, color="steelblue")
    plt.scatter(ek, rho, c="tomato", zorder=999, alpha=0.9, marker="d")
    plt.ylim(0, 1)
    plt.show()


.. image:: imgs/kpm.png

References
----------

    [1] Wang, L.W., 1994. Calculating the density of states and
    optical-absorption spectra of large quantum systems by the plane-wave moments
    method. Physical Review B, 49(15), p.10154.

    [2] Hutchinson, M.F., 1990. A stochastic estimator of the trace of the
    influence matrix for laplacian smoothing splines. Communications in
    Statistics-Simulation and Computation, 19(2), pp.433-450.



Sthocastic Lanczos Quadrature
=============================

Given a semi-positive definite matrix :math:`A \in \mathbb R^{|V|\times|V|}`,
which has the set of eigenvalues given by :math:`\{\lambda_i\}` a trace of
a matrix function is given by

.. math:: 

    \mathrm{tr}(f(A)) = \sum\limits_{i=0}^{|V|} f(\lambda_i)

The methods for calculating such traces functions have a
cubic computational complexity lower bound,  :math:`O(|V|^3)`.
Therefore, it is not feasible for  large networks. One way
to overcome such computational complexity it is use stochastic approximations
combined with a mryiad of another methods
to get the results with enough accuracy and with a small computational cost. 
The methods available in this module uses the Sthocastic Lanczos Quadrature, 
a procedure proposed in the work made by Ubaru, S. et.al. [1] (you need to cite them).


Estrada Index
-------------

.. code-block:: python

    import scipy
    import scipy.sparse
    import igraph as ig
    import numpy as np

    N = 3000
    G = ig.Graph.Erdos_Renyi(N, 3/N) 

.. code-block:: python

    from emate.symmetric.slq import pyslq
    import tensorflow as tf

    def trace_function(eig_vals):
        return tf.exp(eig_vals)

    num_vecs = 100
    num_steps = 50
    approximated_estrada_index, _ = pyslq(L_sparse, num_vecs, num_steps,  trace_function)
    exact_estrada_index =  np.sum(np.exp(vals_laplacian))
    approximated_estrada_index, exact_estrada_index


The above code returns

.. code-block:: bash

    (3058.012, 3063.16457163222)

References
----------

    1 - Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature. 
    SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.

    2 - Hutchinson, M. F. (1990). A stochastic estimator of the trace of the
    influence matrix for laplacian smoothing splines. Communications in
    Statistics-Simulation and Computation, 19(2), 433-450.
