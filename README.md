# ![eMaTe](emate.png)

eMaTe is a python package implemented in tensorflow which the main goal is provide useful methods capable of estimate spectral densities and trace functions of large sparse matrices. 


## Kernel Polynomial Method (KPM)

The Kernel Polynomial Method can estimate the spectral density of large sparse Hermitan matrices with a computaional cost almost linear. This method combines three key ingredients, the Chebyshev expansion, the stochastic trace estimator and kernel smoothing.


### Example

```python
import igraph as ig
import numpy as np
import scipy
from scipy.sparse


L_sparse = scipy.sparse.coo_matrix(L)
g = ig.Graph.Erdos_Renyi(3000, 3/3000)
W = np.array(g.get_adjacency().data, dtype=np.float64)
vals_laplacian = np.linalg.eigvals(L).real
L_sparse = scipy.sparse.coo_matrix(L)
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

### Example

#### Computing the Estrada index

```python
from emate.symmetric.slq import pyslq
import tensorflow as tf

def trace_function(eig_vals):
    return tf.exp(eig_vals)

num_vecs = 100
num_steps = 50
approximated_estrada_index, _ = pyslq(L_sparse, num_vecs, num_steps,  trace_function)
exact_estrada_index =  np.sum(np.exp(vals_laplacian))
approximated_estrada_index, exact_estrada_index
```
The above code returns

```
(3058.012, 3063.16457163222)
```
[[1] Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics-Simulation and Computation, 19(2), 433-450.](https://www.tandfonline.com/doi/abs/10.1080/03610919008812866)

[[2] Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099.](https://epubs.siam.org/doi/abs/10.1137/16M1104974)


## Acknowledgements

This work has been supported also by FAPESP grants  11/50761-2  and  2015/22308-2.   Research  carriedout using the computational resources of the Center forMathematical  Sciences  Applied  to  Industry  (CeMEAI)funded by FAPESP (grant 2013/07375-0).
 

