
#shared research and simulated executions. each public name of a 'subpackage' is re-exported here.
from importlib import import_module as _import_module

__version__ = "0.1.0"

_SUBPACKAGES = "costs", "data", "events", "execution", "models", "portfolio", "reporting", "risk_monitor", "signals", "tax", "validate", "vol_regime"


def _owners(modules):
    #map each 'exported name' to its correct subpackage
    owners = {}
    for sub, module in modules.items():
        for name in module.__all__:
            if name in modules:
                raise ImportError(f"foundation.{sub} exports {name!r}, which would shadow the foundation.{name} subpackage; rename it")
            if name in owners:
                raise ImportError(f"{name!r} is exported by both foundation.{owners[name]} and foundation.{sub}; rename one so neither shadows the other")
            owners[name] = sub
            
    return owners


_EXPORTS = _owners({sub: _import_module(f"{__name__}.{sub}") for sub in _SUBPACKAGES})

#noqa: E402, F401, F403

from .costs import *  
from .data import *  
from .events import *  
from .execution import * 
from .models import *  
from .portfolio import * 
from .reporting import * 
from .risk_monitor import *  
from .signals import *  
from .tax import * 
from .validate import *  
from .vol_regime import *  

__all__ = sorted(_EXPORTS)
