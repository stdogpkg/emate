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
 - name: Institute of physics of São Carlos, SP
   index: 1
date: 01 July 2020
bibliography: paper.bib


---

# Summary

Sparse matrices can represent several phenomena and theoretical structures, from social networks to quantum systems. Therefore, having technical tools capable of extracting information about these kinds of matrices it is essential. One of the most important information about matrices are that related with the spectra of the matrices and functions associated. For example, the spectral proprieties are strongly related with dynamical proprieties occurring in complex networks, can compare graphs and estimating quantum density to just name a few. However, the algorithms to calculate this spectral proprieties has $\mathcal O(n^3)$ complexity, where $n$ it is the size of the matrix. Therefore, having a friendly package which allow this spectral information to be estimated in an efficient way can be helpfully.

The `eMaTe` is an Python package for estimating proprieties of sparse matrices using stochastic approaches using GPU or CPU. The stochastic approach can reduce the computational complexity for almost linear, allowing to estimate the spectral proprieties of large matrices. Until now, `eMaTe` has two main algorithms implemented. The Kernel Polynomial Method (KPM) and the Stochastic Lanczos Quadrature (SLQ). The first can estimate spectral density, functions and trace functions of any Hermitian matrices. The second method was recently proposed to estimate efficiently and in accurate form trace functions from symmetric sparse matrices. We can directly apply both methods in the field of undirected graphs (a.k.a. undirected networks), recently has showed how to use this method in directed graphs... 



# Mathematics

The theory of sthocastic estimation of sparse matrices it is realy heavily. Therefore, we will not  discuss that here. The reader interessed in that should consulte ... for the KPM method, and ... for SLQ method. To understand the powerfully sthocastic trace estimator and  the parameters used in both methods see ....



# Examples

The theory of sthocastic estimation of sparse matrices it is realy heavily. Therefore, we will not  discuss that here. The reader interessed in that should consulte ... for the KPM method, and ... for SLQ method. To understand the powerfully sthocastic trace estimator and  the parameters used in both methods see ....
# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

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
