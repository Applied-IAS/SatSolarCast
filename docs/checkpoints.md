# Pretrained checkpoints

This release provides the CrossViViT baseline and LSTM-att+Sat weights for 3-hour
and 6-hour forecasting. The input and output lengths are equal: 12 or 24 steps,
respectively, at 15-minute resolution. LSTM-att+Sat includes the satellite memory;
the `False1` experiment suffix denotes `multi_channel_input=False` and
`use_satellite_memory=1`.

## Download

**Google Drive upload is pending.** The download URL will be added here and to
[`checkpoints/manifest.json`](../checkpoints/manifest.json) after upload. No
checkpoint files are stored in Git. The archive name is
`SatSolarCast-checkpoints.zip`, approximately 1.24 GB.

Extract the archive into the repository root:

```bash
unzip /path/to/SatSolarCast-checkpoints.zip -d .
sha256sum -c checkpoints/SHA256SUMS
```

| Model | Horizon | Weight file | Configuration |
| --- | --- | --- | --- |
| CrossViViT | 3 hours | `checkpoints/crossvivit_3h.pth` | [JSON](../configs/crossvivit_3h.json) |
| CrossViViT | 6 hours | `checkpoints/crossvivit_6h.pth` | [JSON](../configs/crossvivit_6h.json) |
| LSTM-att+Sat | 3 hours | `checkpoints/lstm-att-sat_3h.pth` | [JSON](../configs/lstm-att-sat_3h.json) |
| LSTM-att+Sat | 6 hours | `checkpoints/lstm-att-sat_6h.pth` | [JSON](../configs/lstm-att-sat_6h.json) |

All four files are plain PyTorch state dictionaries. The CLI checks the SHA-256
checksum and loads weights with `strict=True`. Each CrossViViT weight is about
580 MB; each LSTM-att+Sat weight is about 40 MB. Complete architectures live in
`models/` and can be imported without running a training script. The checked-in
configs, model definitions, and checkpoints must be used together.

The inference copy of CrossViViT generates zero-valued position coordinates on
the patch grid. This fixes a pixel/patch token-count mismatch when masking is
disabled and preserves the original masked forward output for the same inputs
and random state. It changes no checkpoint parameters.

## Inference

Use Python 3.11 and install the versions used for validation:

```bash
python -m pip install -r requirements-inference.txt
```

Prepare normalized HDF5 data using the [preprocessing workflow](../preprocessing/README.md).
The input dataset `arr_0` has shape `(N,T,120,120,5)` with channels
`[vis047,vis086,ir133,sza,insitu]`. The fifth channel contains the station's
normalized irradiance broadcast across the image. `arr_1` is optional for
inference and is used only to save reference targets and compute metrics.
Neither released configuration needs a separate point file during inference;
point files are still generated for compatibility with the training code.

3-hour LSTM-att+Sat:

```bash
python infer.py \
  --config configs/lstm-att-sat_3h.json \
  --input-h5 /path/to/solarcube_hybrid_sat_insitu_test_5.h5 \
  --output-dir outputs/lstm-att-sat_3h_station5
```

6-hour CrossViViT:

```bash
python infer.py \
  --config configs/crossvivit_6h.json \
  --input-h5 /path/to/solarcube_hybrid_sat_insitu_24_test_5.h5 \
  --output-dir outputs/crossvivit_6h_station5
```

Select the other configuration files to run the other model/horizon pairs.
Add `--device cpu` for CPU inference, or `--device cuda:0` to select a GPU.
`--checkpoint /path/to/file.pth` overrides the weight location. Add `--limit 1`
for a quick check. The default batch size is 1 in standard inference.

The output directory contains `predictions.npy` with shape `(N,T,1)` in W/m²,
optional `targets.npy`, and `run.json` recording the full configuration, checksum,
seed, batch size, protocol, environment, and metrics. Use `--units normalized` to
retain normalized values. Physical irradiance is `normalized * 279.49 + 193.44`.
Outputs are not clipped. `pearson_r_squared` matches the squared-correlation
definition in the original evaluation script; it is not the residual-based R².

## Evaluation protocols

The default `--protocol standard` uses `model.eval()` and disables CrossViViT
random masking. LSTM-att+Sat uses its history-only matching encoder (`phase=2`).
Future satellite frames and future ground observations are not needed.

The original experiment testing code had two behaviors that affect predictions:

- CrossViViT called `forward` with its default `mask=True`. Masking remains random
  even when `model.eval()` is used. `--protocol legacy` enables this behavior;
  its default batch size is 32 for CrossViViT.
- LSTM-att+Sat constructed a fresh model without calling `model.eval()` before
  testing. Its BatchNorm layers therefore used batch statistics. `--protocol
  legacy` retains that behavior, with batch size 1. It performs no gradient
  updates and never saves changed running statistics back into the checkpoint.

Choose the protocol explicitly when comparing results. A seeded legacy run does
not guarantee an exact match to saved historical arrays: masking also depends
on batching and the preceding random-number sequence. For the paper's four-site
evaluation, process stations `[5,7,9,18]`, concatenate the predictions and targets
in that order, and compute metrics over all samples and lead times. Do not
average station-level RMSEs. The historical 3-hour aggregate test file also
contained 11 station-11 samples, which are excluded from this four-site metric.

Validation covered strict weight loading and real-data sample inference for all
four weights on CPU. A complete rerun of the paper's test metrics has not been
performed for this release. Standard-mode outputs should not be represented as
verified reproductions of the historical paper metrics.

## Maintainer archive preparation

```bash
python scripts/package_checkpoints.py \
  --source-root /path/to/satsolarcast-crossvivit \
  --output /path/to/SatSolarCast-checkpoints.zip
```

The packaging script verifies all four original weights before writing the
archive. Upload it to Google Drive, set the intended download access, and add
the real share URL above and in `checkpoints/manifest.json`.
