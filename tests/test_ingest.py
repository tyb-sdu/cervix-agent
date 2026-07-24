import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from cervixagent.audit import verify_sealed_directory
from cervixagent.ingest import ingest_engineering_test, verify_ingestion_run
from cervixagent.project import complete_current_step, init_project


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "RDKit is not installed")
class IngestTests(unittest.TestCase):
    def test_ingest_retains_records_and_does_not_filter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            init_project(root, "ingest-test")
            complete_current_step(root, "P1-01", {"test": True})
            input_path = root / "data/processed/test_compounds_500.tsv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["record_id", "source", "source_id", "smiles"])
                writer.writerow(["T-1", "COCONUT", "C-1", "CCO"])
                writer.writerow(["T-2", "LOTUS", "L-1", "OCC"])
                writer.writerow(["T-3", "LOTUS", "L-2", "not-a-smiles"])
                writer.writerow(["T-4", "COCONUT", "C-2", "CCO.[Na+]"])

            result = ingest_engineering_test(root, input_path, label="unit-test")
            self.assertFalse(result["formal_p1_02_complete"])
            self.assertEqual(4, result["counts"]["input_records"])
            self.assertEqual(3, result["counts"]["valid_records"])
            self.assertEqual(1, result["counts"]["invalid_records"])
            self.assertEqual(1, result["counts"]["canonical_duplicate_records"])
            self.assertFalse(result["decisions"]["salt_or_fragment_removal"])
            self.assertFalse(result["decisions"]["michael_acceptor_filter"])
            output_dir = root / result["relative_path"]
            self.assertTrue(verify_sealed_directory(output_dir)["valid"])
            self.assertTrue(verify_ingestion_run(root, result["run_id"])["valid"])
            with (output_dir / "records.tsv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(4, len(rows))
            self.assertEqual("2", rows[3]["fragment_count"])


if __name__ == "__main__":
    unittest.main()
