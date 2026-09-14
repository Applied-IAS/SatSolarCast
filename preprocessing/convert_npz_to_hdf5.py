#!/usr/bin/env python3
"""Convert generated SolarCube `.npz` files to SatSolarCast HDF5 files.

The generated SolarCube files contain `arr_0`, `arr_1`, and `arr_2`.
The released SatSolarCast dataloaders read HDF5 files containing only
`arr_0` and `arr_1`; the sample manifest can optionally be exported as CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument(
        "--station-ids",
        type=int,
        nargs="+",
        default=[5, 7, 9, 18],
        help="Held-out station IDs for per-station test HDF5 files.",
    )
    parser.add_argument(
        "--export-manifest",
        action="store_true",
        help="Export arr_2 sample manifests as CSV files.",
    )
    return parser.parse_args()


def write_h5(npz_path: Path, h5_path: Path, indices: np.ndarray | None = None) -> None:
    data = np.load(npz_path, allow_pickle=True)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as fout:
        for key in ("arr_0", "arr_1"):
            arr = data[key]
            if indices is not None:
                arr = arr[indices]
            arr = np.nan_to_num(arr, nan=0.0)
            fout.create_dataset(key, data=arr)

    print(f"Wrote {h5_path}")


def export_manifest(npz_path: Path, csv_path: Path, indices: np.ndarray | None = None) -> None:
    data = np.load(npz_path, allow_pickle=True)
    manifest = data["arr_2"]
    if indices is not None:
        manifest = manifest[indices]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "year", "start_index"])
        writer.writerows(manifest.tolist())

    print(f"Wrote {csv_path}")


def station_indices(npz_path: Path, station_id: int) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    station_ids = data["arr_2"][:, 0].astype(int)
    return np.where(station_ids == station_id)[0]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_h5 = args.out_dir / f"{args.prefix}_train.h5"
    test_h5 = args.out_dir / f"{args.prefix}_test.h5"
    write_h5(args.train_npz, train_h5)
    write_h5(args.test_npz, test_h5)

    if args.export_manifest:
        export_manifest(args.train_npz, args.out_dir / f"{args.prefix}_train_manifest.csv")
        export_manifest(args.test_npz, args.out_dir / f"{args.prefix}_test_manifest.csv")

    for station_id in args.station_ids:
        idx = station_indices(args.test_npz, station_id)
        out_path = args.out_dir / f"{args.prefix}_test_{station_id}.h5"
        write_h5(args.test_npz, out_path, indices=idx)
        if args.export_manifest:
            export_manifest(
                args.test_npz,
                args.out_dir / f"{args.prefix}_test_{station_id}_manifest.csv",
                indices=idx,
            )


if __name__ == "__main__":
    main()

