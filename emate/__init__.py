from emate import hermitian
from emate import symmetric

from emate import linalg
from emate import utils

try:
    import cupy as cp
except:
    print("Warning: CUPy package not found")

__version__ = "1.1.3"
__license__ = "MIT"
__author__ = "Bruno Messias; Thomas K Peron"
__author_email__ = "messias.physics@gmail.com"
__name__ = "eMaTe"


__all__ = ["hermitian", "symmetric", "linalg", "utils"]
