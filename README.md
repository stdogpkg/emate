# eMaTe

eMaTe is a python package implemented in tensorflow which the main goal is provide useful methods capable of estimate spectral densities and trace functions of large sparse matrices. 


## Kernel Polynomial Method (KPM)

The Kernel Polynomial Method can estimate the spectral density of large sparse Hermitan matrices with a computaional cost almost linear. This method combines three key ingredients, the Chebyscev expansion, the stochastic trace estimator and kernel smoothing.


### Example

```python
import igraph as ig
import numpy as np
from scipy import sparse

g = ig.Graph.Erdos_Renyi(3000, 3/3000)
W = np.array(g.get_adjacency().data, dtype=np.float64)
L_sparse = sparse.coo_matrix(L)
vals_laplacian = np.linalg.eigvals(L).real
```

```python
from emate.hermitian import pykpm
num_moments = 50
num_vecs = 100
extra_points = 10
ek_laplacian, rho_laplacian = pykpm(L_sparse, num_moments, num_vecs, extra_points)
```

## Stochastic Lanczos Quadrature (SLQ)


>The problem of estimating the trace of matrix functions appears in applications ranging from machine learning and scientific computing, to computational biology.[2] 



[[2] Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.](https://epubs.siam.org/doi/abs/10.1137/16M1104974)
