#prevent changed inputs or damaged cache files from displaying stale results
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversal_ml_data import SNAPSHOT, SPEC
from reversal_ml_research import ResearchCache


class ResearchCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        for relative in (SNAPSHOT, SPEC, "src/study.py"):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen fixture")

    def test_input_and_source_changes_require_a_new_cache(self):
        original = ResearchCache(self.project)
        original.get("development", lambda: {"answer": 1})
        (self.project / SNAPSHOT).write_text("changed prices")
        changed = ResearchCache(self.project)
        self.assertNotEqual(original.fingerprint, changed.fingerprint)
        self.assertEqual(changed.get("development", lambda: {"answer": 2}), {"answer": 2})
        (self.project / "src/study.py").write_text("changed model")
        self.assertNotEqual(changed.fingerprint, ResearchCache(self.project).fingerprint)

    def test_damaged_cache_fails_and_explicit_rebuild_recovers(self):
        cache = ResearchCache(self.project)
        cache.get("development", lambda: {"answer": 1})
        (cache.directory / "development.pkl").write_bytes(b"truncated cache")
        with self.assertRaisesRegex(ValueError, "Changed or damaged"):
            cache.get("development", lambda: {"answer": 2})
        self.assertEqual(cache.get("development", lambda: {"answer": 2}, use_cache=False), {"answer": 2})


if __name__ == "__main__":
    unittest.main()
