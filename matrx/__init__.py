# Aliasing imports
from matrx.world_builder import WorldBuilder


__docformat__ = "restructuredtext"


######
# We do this so we are sure everything is imported and thus can be found
# noinspection PyUnresolvedReferences
import pkgutil
import importlib

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    short_name = module_name.split(".")[-1]
    __all__.append(short_name)
    _module = importlib.import_module(module_name)
    globals()[short_name] = _module
######

# Set package attributes
name = "MATRX: Man-Agent Teaming - Rapid Experimentation Software"
