#!/usr/bin/env python3
"""Stream SolarCube raw products into the HDF5 files used by released models."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

CHANNELS = ["vis047", "vis086", "ir133", "sza", "insitu"]
OFFSETS = np.array([0.56, 0.55, 258.55, 56.67, 193.44], dtype=np.float64)
SCALES = np.array([0.39, 0.32, 17.13, 20.19, 279.49], dtype=np.float64)


def sample_manifest(data_home, tiles, years, sequence_length, interval, sitelist):
    rows = []
    for year in years:
        for tile in tiles:
            valid = pd.read_csv(data_home / f"{tile}_{year}_index.csv")[
                CHANNELS
            ].to_numpy()
            valid = (valid > 0).all(axis=1)
            station = int(sitelist.loc[tile, "id"])
            for start in range(0, len(valid) - sequence_length + 1, interval):
                if valid[start : start + sequence_length].all():
                    rows.append((station, str(year), start, int(tile)))
    return pd.DataFrame(rows, columns=["station_id", "year", "start_index", "tile_id"])


def smoothed_insitu(values):
    values = np.asarray(values, dtype=np.float64)
    return np.convolve(
        np.pad(values, (7, 7), mode="edge"), np.ones(15) / 15, mode="valid"
    )[::15]


def load_sequence(data_home, row, length, insitu, product_prefix):
    start = int(row.start_index)
    images = []
    for channel in CHANNELS[:-1]:
        path = data_home / f"{product_prefix}_{row.year}_{row.tile_id}_{channel}.hdf"
        with h5py.File(path, "r") as source:
            dataset = source[channel]
            if dataset.shape[1:] != (120, 120):
                raise ValueError(
                    f"{path}: expected 120x120 images, got {dataset.shape}"
                )
            raw = dataset[start : start + length]
            array = raw.astype(np.float64) * dataset.attrs["scale_factor"]
            array[raw == dataset.attrs["fillvalue"]] = np.nan
            images.append(array)
    values = insitu[int(row.station_id)][start : start + length]
    if len(values) != length or any(len(image) != length for image in images):
        raise ValueError("Sample extends past available observations.")
    images.append(np.broadcast_to(values[:, None, None], (length, 120, 120)))
    return np.stack(images, axis=-1)


def normalize(array, target=False):
    # The original loader cast targets to float32 before normalization, but
    # normalized inputs at the raw-product precision before the final cast.
    if target:
        array = array.astype(np.float32)
    channels = [
        (array[..., i] - float(OFFSETS[i])) / float(SCALES[i]) for i in range(5)
    ]
    result = np.stack(channels, axis=-1).astype(np.float32)
    return np.nan_to_num(result, copy=False, nan=0.0)


def create_arrays(output, count, input_len, output_len):
    for key, length in [("arr_0", input_len), ("arr_1", output_len)]:
        shape = (count, length, 120, 120, 5)
        if count:
            output.create_dataset(
                key,
                shape=shape,
                dtype="float32",
                chunks=(1, 1, 120, 120, 5),
                compression="lzf",
            )
        else:
            output.create_dataset(key, shape=shape, dtype="float32")


def write_points(path, manifest, sitelist):
    points = np.asarray(
        [
            (
                int(sitelist.loc[row.tile_id, "fLine"]),
                int(sitelist.loc[row.tile_id, "fCol"]),
            )
            for row in manifest.itertuples(index=False)
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    with h5py.File(path, "x") as output:
        output.create_dataset("points", data=points)


def write_split(args, config, split, manifest, sitelist, insitu):
    stem = args.out_dir / f"{args.prefix}_{split}"
    input_len, output_len = config["input_len"], config["output_len"]
    with h5py.File(f"{stem}.h5", "x") as output:
        create_arrays(output, len(manifest), input_len, output_len)
        for index, row in enumerate(manifest.itertuples(index=False)):
            array = load_sequence(
                args.data_home, row, input_len + output_len, insitu, args.product_prefix
            )
            output["arr_0"][index] = normalize(array[:input_len])
            output["arr_1"][index] = normalize(array[input_len:], target=True)
            if index % 100 == 0:
                print(f"{split}: {index + 1}/{len(manifest)}", flush=True)
    manifest[["station_id", "year", "start_index"]].to_csv(
        f"{stem}_manifest.csv", index=False
    )
    write_points(f"{stem}_points.h5", manifest, sitelist)


def write_station_splits(args, config, manifest, sitelist):
    with h5py.File(args.out_dir / f"{args.prefix}_test.h5", "r") as source:
        for tile in config["test_tile_list"]:
            station = int(sitelist.loc[tile, "id"])
            indices = np.flatnonzero(manifest.station_id.to_numpy() == station)
            stem = args.out_dir / f"{args.prefix}_test_{station}"
            with h5py.File(f"{stem}.h5", "x") as output:
                create_arrays(
                    output, len(indices), config["input_len"], config["output_len"]
                )
                for destination, index in enumerate(indices):
                    for key in ["arr_0", "arr_1"]:
                        output[key][destination] = source[key][index]
            station_manifest = manifest.iloc[indices]
            station_manifest[["station_id", "year", "start_index"]].to_csv(
                f"{stem}_manifest.csv", index=False
            )
            write_points(f"{stem}_points.h5", station_manifest, sitelist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--product-prefix", choices=["SolarCube", "SolarSat"], default="SolarCube"
    )
    parser.add_argument(
        "--limit", type=int, help="Smoke check only: maximum samples per split."
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = yaml.safe_load(args.cfg.read_text())["dataset"]
    if any(config[key] != CHANNELS for key in ["x_img_types", "y_img_types"]):
        parser.error(f"Expected input and output channel order {CHANNELS}")
    if any(
        config[key]
        for key in [
            "point_based",
            "downscale_s",
            "downscale_t",
            "ignorenan",
            "normalize_max",
        ]
    ):
        parser.error(
            "This release uses full-resolution images, valid sample windows, and standard normalization."
        )
    if not config["normalize_x"] or not config["normalize_y"]:
        parser.error(
            "The pretrained checkpoints require normalized inputs and targets."
        )
    for split in ["train", "test"]:
        if [str(year) for year in config[f"{split}_year_list"]] != ["2018"]:
            parser.error(
                "The released workflow supports the 2018 SolarCube in-situ CSV."
            )
    sitelist = pd.read_csv(args.data_home / "solarcube_sitelist.csv").set_index(
        "tile_id"
    )
    if not sitelist.index.is_unique or not sitelist["id"].is_unique:
        parser.error("Sitelist must map each tile to a unique station ID.")
    insitu_frame = pd.read_csv(
        args.data_home / "solarcube_insitu.csv", low_memory=False
    )
    tiles = config["train_tile_list"] + config["test_tile_list"]
    if set(config["train_tile_list"]) & set(config["test_tile_list"]):
        parser.error("Training and test tiles must be disjoint.")
    insitu = {
        int(sitelist.loc[tile, "id"]): smoothed_insitu(
            insitu_frame[str(int(sitelist.loc[tile, "id"]))]
        )
        for tile in tiles
    }
    stems = [f"{args.prefix}_train", f"{args.prefix}_test"]
    stems += [
        f"{args.prefix}_test_{int(sitelist.loc[tile, 'id'])}"
        for tile in config["test_tile_list"]
    ]
    for stem in stems:
        for suffix in [".h5", "_points.h5", "_manifest.csv"]:
            if (args.out_dir / f"{stem}{suffix}").exists():
                parser.error(
                    "Output files already exist; choose a new output directory."
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ["train", "test"]:
        manifest = sample_manifest(
            args.data_home,
            config[f"{split}_tile_list"],
            config[f"{split}_year_list"],
            config["input_len"] + config["output_len"],
            config["sample_interval"],
            sitelist,
        )
        if args.limit:
            manifest = manifest.iloc[: args.limit].copy()
        summary[split] = manifest.groupby("station_id").size().to_dict()
        write_split(args, config, split, manifest, sitelist, insitu)
        if split == "test":
            write_station_splits(args, config, manifest, sitelist)
    print(f"Sample counts by station: {summary}")


if __name__ == "__main__":
    main()
