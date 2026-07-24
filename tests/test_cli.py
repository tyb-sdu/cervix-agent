import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cervixagent.cli import main


class CliTests(unittest.TestCase):
    def test_workflow_command(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["workflow"])
        self.assertEqual(0, code)
        self.assertIn("phase_1", output.getvalue())
        self.assertIn("phase_3", output.getvalue())

    def test_init_and_status_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(0, main(["init", str(root), "--name", "demo"]))
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["status", str(root)]))
            self.assertIn("demo", output.getvalue())
            self.assertIn("P1-01", output.getvalue())

    def test_data_sources_command(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["data", "sources"])
        self.assertEqual(0, code)
        self.assertIn("pdb_4xr8", output.getvalue())
        self.assertIn("coconut_drug_discovery", output.getvalue())

    def test_ingest_contract_keeps_scientific_filtering_disabled(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["ingest", "contract", "--json"])
        self.assertEqual(0, code)
        self.assertIn('"scientific_filtering_allowed": false', output.getvalue())
        self.assertIn('"ECNPDB"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
