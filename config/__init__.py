# encoding: utf-8
"""Configuration entry point.

``cfg`` is the mutable default configuration.  ``yacs`` is used when it is
installed; otherwise the small fallback in :mod:`config.cfg_node` provides the
same API, so the package runs on a bare environment.
"""

try:
    from yacs.config import CfgNode  # noqa: F401  (re-exported for convenience)
    YACS_AVAILABLE = True
except ImportError:  # pragma: no cover
    from config.cfg_node import CfgNode  # noqa: F401
    YACS_AVAILABLE = False

from .defaults import _C as cfg
from .defaults import _C as cfg_test

__all__ = ["cfg", "cfg_test", "CfgNode", "YACS_AVAILABLE"]
