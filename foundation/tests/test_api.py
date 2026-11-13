#checking that foundation can be imported into the associated projects
import importlib
import pkgutil
import types

import pytest

import foundation


def subpackages():
    return sorted(m.name for m in pkgutil.iter_modules(foundation.__path__) if m.ispkg and not m.name.startswith("_"))


def exported():
    return {sub: importlib.import_module(f"foundation.{sub}") for sub in subpackages()}


def test_import_and_every_subpackage_is_flattened():
    assert foundation.__version__
    assert tuple(subpackages()) == foundation._SUBPACKAGES


@pytest.mark.parametrize("sub", subpackages())
def test_subpackage_names_reachable_as_same_object(sub):
    module = importlib.import_module(f"foundation.{sub}")
    assert module.__doc__ and module.__doc__.strip()
    assert module.__all__ and len(set(module.__all__)) == len(module.__all__)
    for name in module.__all__:
        assert not name.startswith("_")
        assert getattr(foundation, name) is getattr(module, name), name


def test_no_duplicate_names_across_subpackages():
    seen = {}
    for sub, module in exported().items():
        for name in module.__all__:
            assert name not in seen, f"{name} in foundation.{seen[name]} and foundation.{sub}"
            assert name not in foundation._SUBPACKAGES, name
            seen[name] = sub
    for sub in foundation._SUBPACKAGES:
        assert isinstance(getattr(foundation, sub), types.ModuleType)


def test_top_level_all_sorted_and_complete():
    union = {name for module in exported().values() for name in module.__all__}
    assert foundation.__all__ == sorted(foundation.__all__) == sorted(union)
    assert len(foundation.__all__) == len(set(foundation.__all__))


def test_collisions_fail_loudly():
    a = types.SimpleNamespace(__all__=["tranche"])
    b = types.SimpleNamespace(__all__=["tranche"])
    with pytest.raises(ImportError, match="foundation.a and foundation.b"):
        foundation._owners({"a": a, "b": b})
    with pytest.raises(ImportError, match="shadow"):
        foundation._owners({"a": types.SimpleNamespace(__all__=["b"]), "b": b})


def test_live_consumer_names():
    from foundation.portfolio.legacy_rebalance import rebalance_with_costs
    assert foundation.rebalance_with_costs is rebalance_with_costs
    assert isinstance(foundation.CostModel(10), foundation.CostModel)
    assert isinstance(foundation.TrialLog, type)
