---
title: 'eMaTe a python package to efficienty estimate spectra and spectral functions of sparse matrices'
tags:
  - Python
  - physics
  - graphs
  - complex networks
authors:
  - name:Bruno Messias F. Resende^[messias.physics at gmail.com]
    orcid: 0000-0002-7614-1462
    affiliation: "1" 
affiliations:
 - name: Institute of Physics of São Carlos-USP, SP
   index: 1
date: 01 July 2020
bibliography: paper.bib


---

# Summary

Sparse matrices can represent several phenomena and theoretical structures, from social networks to quantum systems. Therefore, having technical tools capable of extracting information about these kinds of matrices it is essential. One of the most important information about matrices are that related with the spectra of the matrices and functions associated. For example, the spectral proprieties are strongly related with dynamical proprieties occurring in complex networks`[@spectra2007]`, that the spectra can  be used to compare graphs and estimating quantum density to just name a few. However, the algorithms to calculate this spectral proprieties has $\mathcal O(n^3)$ complexity, where $n$ it is the size of the matrix. Therefore, having a friendly package which allow this spectral information to be estimated in an efficient way can be helpfully.

The `eMaTe` is a Python package which can estimate spectral proprieties of sparse matrices using stochastic approaches running in GPU or CPU. The stochastic approach can reduce the computational complexity for almost linear, allowing to estimate the spectral density and functions associated of large sparse matrices. Until now, `eMaTe` has two main algorithms implemented. The Kernel Polynomial Method (KPM) and the Stochastic Lanczos Quadrature (SLQ). The first can estimate spectral density, functions and trace functions of any Hermitian matrices`[@kpm2006]`. The second method was recently proposed to estimate efficiently and in accurate form trace functions from symmetric sparse matrices `[@slq2017]`. We can directly apply both methods in the field of undirected graphs (a.k.a. undirected networks), recently has showed how to use this method in directed graphs... 



# Mathematics

The theory of sthocastic estimation of sparse matrices it is realy heavily. Therefore, we will not  discuss that here. The reader interessed in that should consulte `[@kpm2006]` for the KPM method, and `[@slq2017]` for SLQ method. To understand the powerfully sthocastic trace estimator and  the parameters used in both methods see ....



# Examples

# Citations



# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```	

# Acknowledgements

...

# References
