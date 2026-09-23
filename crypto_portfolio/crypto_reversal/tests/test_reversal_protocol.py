"""Protect a declared research specification against accidental replacement."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reversal_protocol


class ProtocolTests(unittest.TestCase):
    def test_first_write_and_repeat_preserve_exact_file_and_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = reversal_protocol.freeze_research_protocol(root)
            path = root / "research/research_protocol_v2.json"
            before, modified_at = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(json.loads(before), first)
            second = reversal_protocol.freeze_research_protocol(root)
            self.assertEqual(second, first)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, modified_at)

    def test_changed_saved_selection_is_rejected_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = reversal_protocol.freeze_research_protocol(root)
            saved["signal"]["selection_fraction"] = .10
            path = root / "research/research_protocol_v2.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "new version"):
                reversal_protocol.freeze_research_protocol(root)
            self.assertEqual(path.read_bytes(), before)

    def test_corrupt_existing_record_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reversal_protocol.freeze_research_protocol(root)
            path = root / "research/research_protocol_v2.json"
            path.write_text('{"unfinished":', encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaises(json.JSONDecodeError):
                reversal_protocol.freeze_research_protocol(root)
            self.assertEqual(path.read_bytes(), before)

    def test_modifying_returned_settings_cannot_change_later_defaults(self):
        original = reversal_protocol.research_protocol()
        changed = reversal_protocol.research_protocol()
        changed["excluded_products"].clear()
        changed["signal"]["formation_days"] = 7
        changed["holdout_start"] = "2026-01-01"
        self.assertEqual(reversal_protocol.research_protocol(), original)

    def test_current_split_and_cost_basis(self):
        spec = reversal_protocol.research_protocol()
        self.assertEqual(spec["development_start"], "2022-01-01")
        self.assertEqual(spec["universe"]["min_assets"], 1)
        self.assertEqual(spec["development_end_exclusive"], "2025-01-01")
        self.assertEqual(spec["holdout_start"], spec["development_end_exclusive"])
        self.assertEqual(spec["holdout_end_exclusive"], "2026-09-01")
        self.assertTrue(spec["bootstrap_phase"].startswith("Holdout only"))
        self.assertIn("absolute net traded notional", spec["transaction_cost_basis"])


if __name__ == "__main__":
    unittest.main()
