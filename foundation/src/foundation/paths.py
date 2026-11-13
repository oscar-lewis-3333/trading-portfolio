
#this repo holds all the studies, but lots of useful folders live outside of this and are imported in. this file gives the locations of such folders

import json
import os
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parents[4]                  
_DEFAULT = "trading_research_notes"
_NAMES_FILE = Path(__file__).resolve().parents[2] / "private_names.json"
_names = None


def research_root(py=None):
    #local research folder used

    env = os.environ.get("TP_RESEARCH_ROOT")
    if env:
        return Path(env).expanduser()
    return Path(py if py is not None else PY_ROOT) / _DEFAULT


RESEARCH_ROOT = research_root()


def _table():
    global _names
    if _names is None:
        try:
            _names = json.loads(_NAMES_FILE.read_text())
        except (OSError, ValueError):
            _names = {}
    return _names


def research(key, py=None):
    #named file/folder inside the research folder

    rel = _table().get("research", {}).get(key)
    return research_root(py) / (rel if rel is not None else f"unavailable/{key}")


def _rel(path, py):
    base = Path(py if py is not None else PY_ROOT)

    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


def research_rel(rest="", py=None):
    #a path inside research folder as the protocols have it recorded.

    root = research_root(py)
    return _rel(root / rest if rest else root, py)


def research_key(key, py=None):
    #named research file as the protocols have it recorded

    return _rel(research(key, py), py)


def name(key):
    #private file name/record key
    return _table().get("names", {}).get(key, f"<{key}>")


class _Text(dict):
    def __missing__(self, key):
        return f"<{key}>"


def project_text(root):
    #projects frozen wording
    
    try:
        return _Text(json.loads((Path(root) / "protocol_text.json").read_text()).get("text", {}))
    except (OSError, ValueError):
        return _Text()
