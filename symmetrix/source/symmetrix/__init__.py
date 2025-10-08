__version__ = "0.0.1"

# TODO: very hacky, to import underscored names
import importlib
_sym = importlib.import_module('.symmetrix', __name__)
_sym.__all__ = [n for n in vars(_sym) if not (n.startswith('__') and n.endswith('__'))]
from .symmetrix import *

from .symmetrix_calc import Symmetrix
