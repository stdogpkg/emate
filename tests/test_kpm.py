from sklearn.neighbors import KernelDensity
from scipy import integrate
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#from .context import emate
import emate

try:
    import cupy as cp
    textCupyImplementation = True
except:
    textCupyImplementation = False


def calc_cupykpm(W, vals, kde):
    import cupy  as cp
    print("\nCUPY version", cp.__version__, "\n")

    num_moments = 50
    num_vecs = 50
    extra_points = 5
    ek, rho = emate.hermitian.cupykpm(
        W.tocsr().astype("complex64"), num_moments, num_vecs, extra_points)

    print("Saving the cupyKPM plot..")
    plt.hist(vals, density=True, bins=100, alpha=.9, color="steelblue")
    plt.scatter(ek.get(), rho.get(), c="tomato", zorder=999, alpha=0.9, marker="d")
    plt.savefig("tests/test_kpm_cupy.png", filetype="png")

    log_dens = kde.score_samples(ek.get()[:, np.newaxis])

    return integrate.simps(ek.get(), np.abs(rho.get()-np.exp(log_dens))) < 0.01

def calc_tfkpm(W, vals, kde):
    import tensorflow as tf
    print("\nTF version", tf.__version__, "\n")

    num_moments = 100
    num_vecs = 100
    extra_points = 5
    ek, rho = emate.hermitian.tfkpm(
            W.tocsr().astype("complex64"), num_moments, num_vecs, extra_points,
            device="/CPU:0")

    print("Saving the tfKPM plot..")
    plt.hist(vals, density=True, bins=100, alpha=.9, color="steelblue")
    plt.scatter(ek, rho, c="tomato", zorder=999, alpha=0.9, marker="d")
    plt.savefig("tests/test_kpm_tf.png", filetype="png")

    log_dens = kde.score_samples(ek[:, np.newaxis])

    return integrate.simps(ek, np.abs(rho-np.exp(log_dens))) < 0.01

def test_tfkpm():
    n = 1000
    g = nx.erdos_renyi_graph(n , 3/n)
    W = nx.adjacency_matrix(g)

    vals  = np.linalg.eigvals(W.todense()).real

    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(vals[:, np.newaxis])

    assert calc_tfkpm(W, vals, kde)

def test_cupykpm():
    n = 1000
    g = nx.erdos_renyi_graph(n , 3/n)
    W = nx.adjacency_matrix(g)

    vals  = np.linalg.eigvals(W.todense()).real

    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(vals[:, np.newaxis])

    if textCupyImplementation:
        assert calc_cupykpm(W, vals, kde)

