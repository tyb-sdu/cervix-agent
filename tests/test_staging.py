import json
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

from cervixagent.data import sha256_file
from cervixagent.project import complete_current_step, init_project
from cervixagent.staging import stage_public_snapshots, verify_staging_run


class StagingTests(unittest.TestCase):
    def test_two_source_staging_retains_invalid_and_duplicate_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            init_project(root, "staging-test")
            complete_current_step(root, "P1-01", {"test": True})

            coconut = root / "data/raw/coconut/COCONUT_2024_08_DrugDiscovery.tsv.zip"
            coconut.parent.mkdir(parents=True, exist_ok=True)
            coconut_table = (
                "COCONUT_ID\tSMILES\n"
                "C-1\tCCO\n"
                "C-2\tnot-a-smiles\n"
                "C-3\tCCO.[Na+]\n"
            )
            with zipfile.ZipFile(coconut, "w") as archive:
                archive.writestr("COCONUT_2024_08_DrugDiscovery.tsv", coconut_table)

            lotus = root / "data/raw/lotus/LOTUS_DB.smi"
            lotus.parent.mkdir(parents=True, exist_ok=True)
            lotus.write_text(
                "OCC\tL-1\nCCC\tL-2\n[Na+].CCO\tL-3\n",
                encoding="utf-8",
            )
            download_manifest = {
                "schema_version": 1,
                "files": {
                    "coconut_drug_discovery": {"sha256": sha256_file(coconut)},
                    "lotus_smiles": {"sha256": sha256_file(lotus)},
                },
            }
            (root / "data/raw/download_manifest.json").write_text(
                json.dumps(download_manifest), encoding="utf-8"
            )

            result = stage_public_snapshots(root, label="unit-test", batch_size=100)
            self.assertFalse(result["formal_p1_02_complete"])
            self.assertEqual(6, result["counts"]["input_records"])
            self.assertEqual(5, result["counts"]["valid_records"])
            self.assertEqual(1, result["counts"]["invalid_records"])
            self.assertEqual(2, result["counts"]["canonical_duplicate_records"])
            self.assertEqual(3, result["counts"]["unique_valid_structures"])
            self.assertFalse(result["decisions"]["pains_filter"])
            verified = verify_staging_run(root, result["run_id"])
            self.assertTrue(verified["valid"])
            self.assertEqual(0, verified["database_checks"]["orphan_duplicate_references"])

            database = root / result["relative_path"] / "compounds.sqlite"
            connection = sqlite3.connect(database)
            try:
                record_count = connection.execute(
                    "SELECT COUNT(*) FROM compound_record"
                ).fetchone()[0]
                invalid_count = connection.execute(
                    "SELECT COUNT(*) FROM compound_record WHERE validation_status='invalid'"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(6, record_count)
            self.assertEqual(1, invalid_count)


if __name__ == "__main__":
    unittest.main()
