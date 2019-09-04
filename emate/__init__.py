"""
emate package
=============

Modules for Matrix Functions Estimators
---------------------------------------

.. toctree::
  :maxdepth: 1

  emate.hermitian
  emate.symmetric


Subpackages
-----------

.. toctree::
  :maxdepth: 1

  emate.linalg
  emate.utils

"""
from emate import hermitian
from emate import symmetric

from emate import linalg
from emate import utils
__version__ = "1.0.2"
__license__ = ""
__author__ = "Bruno Messias; Thomas K Peron"
__author_email__ = "messias.physics@gmail.com"
__name__ = "eMaTe"


__all__ = ["hermitian", "symmetric", "linalg", "utils"]
