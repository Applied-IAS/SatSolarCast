# SatSolarCast Data Preprocessing

This directory documents and tidies the preprocessing pipeline used to prepare
the HDF5 files consumed by the SatSolarCast LSTM-att experiments.

The full processed HDF5 files are large because they contain full satellite
image sequences. For reproduction, we recommend sharing these scripts and the
sample manifests so users can regenerate the files from the public SolarCube
release.

## Recommended: stream raw SolarCube data to HDF5

Use `prepare_hdf5.py` for the released checkpoints. It writes one sample at a
time, broadcasts each station's in-situ sequence across the fifth image channel,
and generates the data, point files, per-station splits, and CSV sample manifests
together. It avoids loading an entire image-sequence dataset into memory.

Install the preprocessing dependencies:

```bash
python -m pip install -r requirements-preprocessing.txt
```

Obtain the original data from the [SolarCube project](https://github.com/Ruohan-Li/SolarCube).
The input directory must contain:

```text
solarcube_sitelist.csv
solarcube_insitu.csv
1_2018_index.csv                  # one index CSV per configured tile
SolarCube_2018_1_vis047.hdf        # one product per tile and image channel
SolarCube_2018_1_vis086.hdf
SolarCube_2018_1_ir133.hdf
SolarCube_2018_1_sza.hdf
...
```

Each product must contain a dataset named after its channel with shape
`(time,120,120)` and `scale_factor` and `fillvalue` attributes. The index CSVs
contain the five channel validity columns. The sitelist maps `tile_id` to `id`,
`fLine`, and `fCol`. The in-situ CSV contains one column per station ID at
1-minute resolution. This release supports the 2018 task configs below.
For products named `SolarSat_...hdf`, add `--product-prefix SolarSat`.

3-hour task:

```bash
python preprocessing/prepare_hdf5.py \
  --data-home /path/to/solarcube/raw \
  --cfg preprocessing/configs/satsolarcast_3h.yaml \
  --out-dir /path/to/Solar_Radiance_MultiModal \
  --prefix solarcube_hybrid_sat_insitu
```

6-hour task:

```bash
python preprocessing/prepare_hdf5.py \
  --data-home /path/to/solarcube/raw \
  --cfg preprocessing/configs/satsolarcast_6h.yaml \
  --out-dir /path/to/Solar_Radiance_MultiModal_24 \
  --prefix solarcube_hybrid_sat_insitu_24
```

Add `--limit 2` and use a separate output directory for a small smoke check.
The script refuses to overwrite existing output files. A limited run selects
the first samples in tile order and can therefore produce empty files for
later test stations; it is not a full evaluation dataset.

The full output includes `{prefix}_{train,test}.h5`, corresponding
`*_points.h5` and `*_manifest.csv` files, and `test_{5,7,9,18}` variants of each.
Input/target channels are `[vis047,vis086,ir133,sza,insitu]`. Normalization uses
`(value - offset) / scale`:

| Channel | Offset | Scale |
| --- | --- | --- |
| vis047 | 0.56 | 0.39 |
| vis086 | 0.55 | 0.32 |
| ir133 | 258.55 | 17.13 |
| sza | 56.67 | 20.19 |
| insitu | 193.44 | 279.49 |

Satellite fill values become NaN before normalization. NaN is replaced with
zero after normalization; infinities follow NumPy's `nan_to_num` behavior to
match the historical conversion. A 15-minute centered moving average with
seven edge-padded values at each end is used for ground observations, followed
by selecting every 15th value. In-situ is broadcast over all 120x120 pixels.
Only windows whose index values are positive for every channel and time step
are retained. The 3-hour task uses an 8-step (2-hour) window stride, and the
6-hour task uses a 16-step (4-hour) stride. These strides do not change the
15-minute spacing within each sequence.

The local 2018 index files yield the following full-run sample counts:

| Task | Train | Four-site test |
| --- | --- | --- |
| 3 hours | 10,563 | 3,426 |
| 6 hours | 445 | 233 |

Historical 3-hour aggregate files also contained 11 samples from station 11,
giving 3,437 test samples. These are outside the reported four-site aggregate.
The configs here intentionally generate the four-site test set.

## Legacy NPZ conversion route

The remaining sections describe compatibility with previously generated NPZ
files. The upstream generator requires adaptation to broadcast in-situ across
the image before stacking channels when `point_based=False`; its unmodified
loader can raise an incompatible-shape error. Use the streaming workflow above
to avoid that dependency. The legacy converter loads full NPZ arrays and makes
additional copies: a single 3-hour training array is approximately 34 GiB.

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
