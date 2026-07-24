import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cervixagent.audit import create_p1_01_baseline, verify_audit_run
from cervixagent.doctor import Check
from cervixagent.project import init_project, load_project


class AuditTests(unittest.TestCase):
    def _ready_checks(self):
        return [
            Check("Python", "available", "3.12", "core"),
            Check("Disk", "available", "100 GB", "core"),
            Check("Git", "available", "git", "development"),
            Check("RDKit", "available", "rdkit", "phase_1_filtering"),
        ]

    def test_baseline_is_sealed_verified_and_advances_one_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            init_project(root, "audit-test")
            with patch("cervixagent.audit.run_checks", return_value=self._ready_checks()):
                result = create_p1_01_baseline(root, label="unit-test")
            self.assertTrue(result["ready_for_p1_02"])
            self.assertTrue(result["advanced_to_p1_02"])
            self.assertEqual("P1-02", load_project(root)["current_step"])
            verification = verify_audit_run(root, result["run_id"])
            self.assertTrue(verification["valid"])

            environment = root / result["relative_path"] / "environment.json"
            environment.write_text("{}\n", encoding="utf-8")
            tampered = verify_audit_run(root, result["run_id"])
            self.assertFalse(tampered["valid"])
            self.assertTrue(tampered["errors"])


if __name__ == "__main__":
    unittest.main()
