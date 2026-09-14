#!/usr/bin/env python3
"""Create station point-location HDF5 files for SatSolarCast dataloaders."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--sitelist", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument(
        "--station-ids",
        type=int,
        nargs="+",
        default=[5, 7, 9, 18],
        help="Held-out station IDs for per-station point files.",
    )
    return parser.parse_args()


def load_station_points(sitelist_path: Path) -> dict[int, tuple[int, int]]:
    stations = pd.read_csv(sitelist_path, low_memory=False)
    points: dict[int, tuple[int, int]] = {}
    for row in stations.itertuples(index=False):
        station_id = int(getattr(row, "id"))
        fline = int(getattr(row, "fLine"))
        fcol = int(getattr(row, "fCol"))
        points[station_id] = (fline, fcol)
    return points


def points_from_manifest(npz_path: Path, station_points: dict[int, tuple[int, int]]) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    station_ids = data["arr_2"][:, 0].astype(int)
    points = [station_points[station_id] for station_id in station_ids]
    return np.asarray(points, dtype=np.int64)


def write_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("points", data=points)
    print(f"Wrote {path} {points.shape}")


def main() -> None:
    args = parse_args()
    station_points = load_station_points(args.sitelist)

    train_points = points_from_manifest(args.train_npz, station_points)
    test_points = points_from_manifest(args.test_npz, station_points)

    write_points(args.out_dir / f"{args.prefix}_train_points.h5", train_points)
    write_points(args.out_dir / f"{args.prefix}_test_points.h5", test_points)

    test_data = np.load(args.test_npz, allow_pickle=True)
    test_station_ids = test_data["arr_2"][:, 0].astype(int)

    for station_id in args.station_ids:
        indices = np.where(test_station_ids == station_id)[0]
        station_points_arr = test_points[indices]
        write_points(
            args.out_dir / f"{args.prefix}_test_{station_id}_points.h5",
            station_points_arr,
        )


if __name__ == "__main__":
    main()

