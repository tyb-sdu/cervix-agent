import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from cervixagent.data import build_test_dataset, download_url, sha256_file
from cervixagent.project import init_project


class DataTests(unittest.TestCase):
    def test_download_url_uses_atomic_verified_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.txt"
            source.write_bytes(b"public test data\n")
            destination = root / "nested/destination.txt"
            digest, byte_count, downloaded = download_url(source.as_uri(), destination)
            self.assertTrue(downloaded)
            self.assertEqual(source.stat().st_size, byte_count)
            self.assertEqual(sha256_file(source), digest)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertFalse((root / "nested/destination.txt.part").exists())

    def test_build_traceable_balanced_test_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            init_project(root, "test")
            lotus = root / "data/raw/lotus/LOTUS_DB.smi"
            lotus.parent.mkdir(parents=True, exist_ok=True)
            lotus.write_text(
                "\n".join(f"CCO{'C' * index}\tLTS-{index}" for index in range(20))
                + "\n",
                encoding="utf-8",
            )
            coconut = root / "data/raw/coconut/COCONUT_2024_08_DrugDiscovery.tsv.zip"
            coconut.parent.mkdir(parents=True, exist_ok=True)
            table = "coconut_id\tcanonical_smiles\n" + "\n".join(
                f"CNP-{index}\tNCC{'C' * index}" for index in range(20)
            )
            with zipfile.ZipFile(coconut, "w") as archive:
                archive.writestr("drug_discovery.tsv", table)

            manifest = build_test_dataset(root, size=20, seed=7)
            self.assertEqual(20, manifest["record_count"])
            self.assertEqual({"COCONUT": 10, "LOTUS": 10}, manifest["source_counts"])
            output = root / manifest["output"]["relative_path"]
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(20, len(rows))
            self.assertEqual(20, len({row["smiles"] for row in rows}))
            self.assertTrue(output.with_suffix(".manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
