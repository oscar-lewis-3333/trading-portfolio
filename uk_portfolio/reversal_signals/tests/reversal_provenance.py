

#ensuring strategy code performs correctly
import ast
import hashlib
from pathlib import Path
from zipfile import ZipFile


REMOVED_FUNCTIONS = {
    "reversal_backtest.py": {"build_drifted_weights"},
    "reversal_data.py": {"fetch_reversal_panel"},
    "reversal_features.py": {
        "build_market_model_coefficients", "build_residual_reversal_features",
    },
    "reversal_validation.py": {
        "build_basket_outcomes", "summarise_basket_comparison", "add_basket_costs",
    },
}
SOURCE_ARCHIVE = "data/source_cleanup_2026_09_15/original_sources.zip"


def _pruned_source(original, removed):

    lines = original.decode("utf-8").splitlines(keepends=True)
    nodes = [node for node in ast.parse(original).body
             if isinstance(node, ast.FunctionDef) and node.name in removed]
    if {node.name for node in nodes} != set(removed):
        raise ValueError("The source archive lacks a declared retired function")
    for node in reversed(nodes):
        start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
        while start and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        end = node.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        del lines[start:end]
    return "".join(lines).encode("utf-8")


def _code_signature(source):

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def verify_frozen_source(path, original_digest):

    path = Path(path)
    current = path.read_bytes()
    if hashlib.sha256(current).hexdigest() == original_digest:
        return
    if path.parent.name != "src":
        raise ValueError(f"Frozen file changed: {path}")
    with ZipFile(path.parent.parent / SOURCE_ARCHIVE) as archive:
        original = archive.read(path.name)
    if hashlib.sha256(original).hexdigest() != original_digest:
        raise ValueError(f"Frozen source archive changed: {path.name}")
    removed = REMOVED_FUNCTIONS.get(path.name)
    expected = _pruned_source(original, removed) if removed else original
    if current == expected:
        return
    if _code_signature(current) != _code_signature(expected):
        raise ValueError(f"Frozen strategy logic changed: {path.name}")
