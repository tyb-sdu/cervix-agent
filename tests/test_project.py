import json
import tempfile
import unittest
from pathlib import Path

from cervixagent.config import load_workflow, workflow_checksum
from cervixagent.project import PROJECT_DIRS, init_project, load_project


class ProjectTests(unittest.TestCase):
    def test_workflow_has_three_locked_phases(self):
        workflow = load_workflow()
        self.assertTrue(workflow["locked"])
        self.assertEqual(3, len(workflow["phases"]))
        self.assertFalse(workflow["policy"]["protocol_changes_by_agent"])

    def test_fixed_screening_parameters_are_present(self):
        parameters = load_workflow()["fixed_parameters"]
        self.assertEqual(500, parameters["filters"]["mw_max"])
        self.assertEqual(20, parameters["experimental_candidates"])
        self.assertEqual(50, parameters["md"]["duration_ns"])

    def test_initialization_creates_lock_and_runtime_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            state = init_project(root, "test-study")
            self.assertEqual("initialized", state["status"])
            self.assertTrue((root / ".cervixagent/workflow.lock.json").exists())
            for relative in PROJECT_DIRS:
                self.assertTrue((root / relative).is_dir(), relative)

            loaded = load_project(root)
            workflow = json.loads(
                (root / ".cervixagent/workflow.lock.json").read_text(encoding="utf-8")
            )
            self.assertEqual(workflow_checksum(workflow), loaded["workflow_sha256"])


if __name__ == "__main__":
    unittest.main()

