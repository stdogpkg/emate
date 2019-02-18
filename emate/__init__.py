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
from . import hermitian
from . import symmetric

from . import linalg
from . import utils
__version__ = "0.0.1"
__license__ = ""
__author__ = "Bruno Messias"
__author_email__ = "messias.physics@gmail.com"
__name__ = "eMaTe"


__all__ = ["hermitian", "symmetric", "linalg", "utils"]
