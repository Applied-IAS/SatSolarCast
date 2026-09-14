# SatSolarCast Data Preprocessing

This directory documents and tidies the preprocessing pipeline used to prepare
the HDF5 files consumed by the SatSolarCast LSTM-att experiments.

The full processed HDF5 files are large because they contain full satellite
image sequences. For reproduction, we recommend sharing these scripts and the
sample manifests so users can regenerate the files from the public SolarCube
release.

## Pipeline Overview

1. Generate SolarCube `.npz` files from the raw SolarCube HDF products.
2. Convert the generated `.npz` arrays to HDF5 files used by the training code.
3. Generate station point-location HDF5 files from the sample manifest.
4. Optionally export/check the sample manifest for reproducibility.

The generated `.npz` files contain:

- `arr_0`: input sequences, shape `(N, input_len, H, W, C)`
- `arr_1`: output sequences, shape `(N, output_len, H, W, C)`
- `arr_2`: sample manifest rows `(station_id, year, start_index)`

The training scripts use HDF5 files containing `arr_0` and `arr_1`, plus
separate `*_points.h5` files containing station pixel coordinates.

## Ground In-situ Processing

In the SolarCube dataloader, the original ground-based irradiance measurements
are read from `solarcube_insitu.csv` at 1-minute resolution. They are smoothed
with a 15-minute moving-average window and then downsampled by taking every
15th value. This creates a 15-minute in-situ sequence aligned with the satellite
observations.

Equivalent code in the original loader:

```python
window = np.ones(15) / 15
convolution = np.convolve(padded_swin, window, mode="valid")
results = convolution[::15]
```

## Task Configurations

The main settings are provided in:

- `configs/satsolarcast_3h.yaml`
- `configs/satsolarcast_6h.yaml`

The station split is station-based:

- training stations: `[1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 19]`
- reported test stations: PSU, TBL, LRC, SAP = `[5, 7, 9, 18]`

The released LSTM-att+SatSolarCast evaluation aggregates only these four
stations. Some older SolarCube configs include station 11, but it is not used
for the reported four-site evaluation.

## Generate `.npz` Files

Use the SolarCube data generator with the corresponding config. Example:

```bash
PYTHONPATH=/path/to/SolarCube-master \
python /path/to/SolarCube-master/solarcube/data_generator.py \
  --save_path /path/to/output/solarcube_hybrid_sat_insitu \
  --cfg preprocessing/configs/satsolarcast_3h.yaml
```

For the 6-hour task:

```bash
PYTHONPATH=/path/to/SolarCube-master \
python /path/to/SolarCube-master/solarcube/data_generator.py \
  --save_path /path/to/output/solarcube_hybrid_sat_insitu_24 \
  --cfg preprocessing/configs/satsolarcast_6h.yaml
```

These commands produce files such as:

```text
solarcube_hybrid_sat_insitu_train.npz
solarcube_hybrid_sat_insitu_test.npz
solarcube_hybrid_sat_insitu_24_train.npz
solarcube_hybrid_sat_insitu_24_test.npz
```

With the configs in this directory, the generated test `.npz` files contain
only the four reported test stations `[5, 7, 9, 18]`.

## Convert `.npz` to HDF5

3-hour:

```bash
python preprocessing/convert_npz_to_hdf5.py \
  --train-npz /path/to/solarcube_hybrid_sat_insitu_train.npz \
  --test-npz /path/to/solarcube_hybrid_sat_insitu_test.npz \
  --out-dir /path/to/Solar_Radiance_MultiModal \
  --prefix solarcube_hybrid_sat_insitu \
  --station-ids 5 7 9 18 \
  --export-manifest
```

6-hour:

```bash
python preprocessing/convert_npz_to_hdf5.py \
  --train-npz /path/to/solarcube_hybrid_sat_insitu_24_train.npz \
  --test-npz /path/to/solarcube_hybrid_sat_insitu_24_test.npz \
  --out-dir /path/to/Solar_Radiance_MultiModal_24 \
  --prefix solarcube_hybrid_sat_insitu_24 \
  --station-ids 5 7 9 18 \
  --export-manifest
```

This script replaces `NaN` values with zero to match the HDF5 files used by the
released training scripts.

## Generate Point Files

3-hour:

```bash
python preprocessing/make_point_files.py \
  --train-npz /path/to/solarcube_hybrid_sat_insitu_train.npz \
  --test-npz /path/to/solarcube_hybrid_sat_insitu_test.npz \
  --sitelist /path/to/solarcube_sitelist.csv \
  --out-dir /path/to/Solar_Radiance_MultiModal \
  --prefix solarcube_hybrid_sat_insitu \
  --station-ids 5 7 9 18
```

6-hour:

```bash
python preprocessing/make_point_files.py \
  --train-npz /path/to/solarcube_hybrid_sat_insitu_24_train.npz \
  --test-npz /path/to/solarcube_hybrid_sat_insitu_24_test.npz \
  --sitelist /path/to/solarcube_sitelist.csv \
  --out-dir /path/to/Solar_Radiance_MultiModal_24 \
  --prefix solarcube_hybrid_sat_insitu_24 \
  --station-ids 5 7 9 18
```

The output point files contain a dataset named `points` with shape `(N, 2)`.
Each row stores the station pixel coordinate copied from `fLine` and `fCol` in
`solarcube_sitelist.csv`, preserving the convention used by the released
dataloader.

## Inspect a Generated `.npz`

```bash
python preprocessing/inspect_npz.py \
  --npz /path/to/solarcube_hybrid_sat_insitu_test.npz
```

This prints array shapes, station counts, and the first/last manifest rows
without loading the large image arrays fully into memory.
