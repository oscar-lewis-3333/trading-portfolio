

#the projects now sit inside group folders, so trading_portfolio.reversal_signals.* no longer
#resolves from the repository parent alone. reversal_preparation.py is hash-pinned by
#prepared_manifest_v1.json and must not be edited, so the package path is widened here instead.
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
GROUP_ROOT = PROJECT.parent
REPO_ROOT = GROUP_ROOT.parent

if str(REPO_ROOT.parent) not in sys.path:
    sys.path.append(str(REPO_ROOT.parent))

import trading_portfolio

if str(GROUP_ROOT) not in list(trading_portfolio.__path__):
    #a plain list survives sys.path changes; a namespace path is recalculated and would drop this
    trading_portfolio.__path__ = [*trading_portfolio.__path__, str(GROUP_ROOT)]
