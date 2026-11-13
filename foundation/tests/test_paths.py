import json

import foundation.paths as FP


def test_research_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TP_RESEARCH_ROOT", str(tmp_path))
    assert FP.research_root() == tmp_path
    assert FP.research_root(py="/elsewhere") == tmp_path


def test_research_root_defaults_beside_the_given_root(monkeypatch, tmp_path):
    monkeypatch.delenv("TP_RESEARCH_ROOT", raising=False)
    root = FP.research_root(py=tmp_path)
    assert root.parent == tmp_path and root.name


def test_missing_names_give_placeholders(monkeypatch, tmp_path):
    monkeypatch.setattr(FP, "_names", {})
    monkeypatch.setenv("TP_RESEARCH_ROOT", str(tmp_path))
    
    assert FP.name("no.such.key") == "<no.such.key>"
    assert FP.research("no.such.key") == tmp_path / "unavailable" / "no.such.key"


def test_named_files_resolve_inside_the_research_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(FP, "_names", {"research": {"spec.x": "specs/x.md"}, "names": {"key.y": "value"}})
    monkeypatch.setenv("TP_RESEARCH_ROOT", str(tmp_path / "notes"))

    assert FP.research("spec.x") == tmp_path / "notes" / "specs" / "x.md"
    assert FP.research_key("spec.x", py=tmp_path) == "notes/specs/x.md"
    assert FP.research_rel("a/b.json", py=tmp_path) == "notes/a/b.json"
    assert FP.name("key.y") == "value"


def test_project_text_reads_the_text_table(tmp_path):
    (tmp_path / "protocol_text.json").write_text(json.dumps({"text": {"a.1": "frozen wording"}}))
    t = FP.project_text(tmp_path)
    assert t["a.1"] == "frozen wording" and t["a.2"] == "<a.2>"
    assert FP.project_text(tmp_path / "missing")["a.1"] == "<a.1>"
