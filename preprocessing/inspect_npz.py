#!/usr/bin/env python3
"""Inspect generated SolarCube `.npz` files without loading image arrays fully."""

from __future__ import annotations

import argparse
import collections
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    return parser.parse_args()


def npy_headers(npz_path: Path) -> list[tuple[str, tuple[int, ...], np.dtype]]:
    headers: list[tuple[str, tuple[int, ...], np.dtype]] = []
    with zipfile.ZipFile(npz_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            with zf.open(name) as f:
                version = npy_format.read_magic(f)
                if version == (1, 0):
                    shape, _fortran, dtype = npy_format.read_array_header_1_0(f)
                else:
                    shape, _fortran, dtype = npy_format.read_array_header_2_0(f)
            headers.append((name[:-4], shape, dtype))
    return headers


def main() -> None:
    args = parse_args()
    print(args.npz)
    for key, shape, dtype in npy_headers(args.npz):
        print(f"{key}: shape={shape}, dtype={dtype}")

    data = np.load(args.npz, allow_pickle=True)
    if "arr_2" not in data.files:
        return

    manifest = data["arr_2"]
    print(f"manifest first rows: {manifest[:3].tolist()}")
    print(f"manifest last rows: {manifest[-3:].tolist()}")
    counts = collections.Counter(manifest[:, 0].astype(int).tolist())
    print(f"station counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

