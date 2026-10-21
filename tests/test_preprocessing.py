import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingWorkflowTest(unittest.TestCase):
    def test_raw_products_to_aligned_station_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, output = root / "raw", root / "processed"
            raw.mkdir()
            config = yaml.safe_load(
                (ROOT / "preprocessing/configs/satsolarcast_3h.yaml").read_text()
            )
            config["dataset"].update(train_tile_list=[1], test_tile_list=[5])
            cfg = root / "task.yaml"
            cfg.write_text(yaml.safe_dump(config))
            pd.DataFrame(
                {"tile_id": [1, 5], "id": [1, 5], "fLine": [3, 7], "fCol": [9, 11]}
            ).to_csv(raw / "solarcube_sitelist.csv", index=False)
            ground = {
                str(station): np.arange(480, dtype=float) + station * 100
                for station in [1, 5]
            }
            pd.DataFrame(ground).to_csv(raw / "solarcube_insitu.csv", index=False)
            channels = ["vis047", "vis086", "ir133", "sza", "insitu"]
            for station in [1, 5]:
                pd.DataFrame({ch: np.ones(32) for ch in channels}).to_csv(
                    raw / f"{station}_2018_index.csv", index=False
                )
                for channel in channels[:-1]:
                    with h5py.File(
                        raw / f"SolarCube_2018_{station}_{channel}.hdf", "w"
                    ) as handle:
                        data = np.full((32, 120, 120), 100, dtype=np.int16)
                        data[0, 0, 0] = -999
                        dataset = handle.create_dataset(channel, data=data)
                        dataset.attrs["fillvalue"] = -999
                        dataset.attrs["scale_factor"] = 0.01
            command = [
                sys.executable,
                str(ROOT / "preprocessing/prepare_hdf5.py"),
                "--data-home",
                str(raw),
                "--cfg",
                str(cfg),
                "--out-dir",
                str(output),
                "--prefix",
                "sample",
            ]
            run = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            with (
                h5py.File(output / "sample_test.h5") as combined,
                h5py.File(output / "sample_test_5.h5") as station,
            ):
                self.assertEqual(combined["arr_0"].shape, (2, 12, 120, 120, 5))
                for key in ["arr_0", "arr_1"]:
                    np.testing.assert_array_equal(combined[key][:], station[key][:])
                sample = combined["arr_0"][0]
                self.assertEqual(sample[0, 0, 0, 0], 0)  # Fill becomes normalized zero.
                self.assertAlmostEqual(
                    float(sample[0, 1, 1, 0]), (1 - 0.56) / 0.39, places=6
                )
                centered_average = np.mean(
                    np.r_[np.repeat(500.0, 7), np.arange(500.0, 508.0)]
                )
                expected_ground = (centered_average - 193.44) / 279.49
                np.testing.assert_allclose(
                    sample[0, :, :, 4], expected_ground, rtol=1e-6
                )
            with h5py.File(output / "sample_test_5_points.h5") as points:
                np.testing.assert_array_equal(points["points"][:], [[7, 11], [7, 11]])
            manifest = pd.read_csv(output / "sample_test_5_manifest.csv")
            self.assertEqual(manifest.station_id.tolist(), [5, 5])
            self.assertEqual(manifest.start_index.tolist(), [0, 8])
            repeat = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("already exist", repeat.stderr)


class ReleaseIntegrityTest(unittest.TestCase):
    def test_wrong_weight_is_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / "wrong.pth"
            invalid.write_bytes(b"not the released weight")
            command = [
                sys.executable,
                str(ROOT / "infer.py"),
                "--config",
                str(ROOT / "configs/lstm-att-sat_3h.json"),
                "--checkpoint",
                str(invalid),
                "--input-h5",
                str(root / "unused.h5"),
                "--output-dir",
                str(root / "output"),
            ]
            run = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("SHA-256 does not match", run.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
