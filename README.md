# SatSolarCast

Code for **SatSolarCast: A Flexible Framework for Multimodal Solar Irradiance
Forecasting via Memory-Alignment Learning** (AAAI 2026).

## Pretrained models

Checkpoint configurations and an inference-only entry point are provided for:

| Model | 3-hour forecast | 6-hour forecast |
| --- | --- | --- |
| CrossViViT baseline | [Configuration](configs/crossvivit_3h.json) | [Configuration](configs/crossvivit_6h.json) |
| LSTM-att+Sat | [Configuration](configs/lstm-att-sat_3h.json) | [Configuration](configs/lstm-att-sat_6h.json) |

**The Google Drive weight upload is pending.** See the
[checkpoint download and inference guide](docs/checkpoints.md) for archive
layout, SHA-256 checksums, model settings, and evaluation protocols. Weights are
hosted separately from this Git repository.

## Quick start

Use Python 3.11. Install inference dependencies, download the checkpoint archive
once available, and extract it into the repository root:

```bash
python -m pip install -r requirements-inference.txt
unzip /path/to/SatSolarCast-checkpoints.zip -d .
sha256sum -c checkpoints/SHA256SUMS
```

Prepare data using the [preprocessing instructions](preprocessing/README.md).
Run a pretrained model without retraining:

```bash
python infer.py \
  --config configs/lstm-att-sat_3h.json \
  --input-h5 /path/to/solarcube_hybrid_sat_insitu_test_5.h5 \
  --output-dir outputs/lstm-att-sat_3h_station5
```

Select any of the four configuration files above. Predictions are saved in
W/m² together with run settings and optional reference targets and metrics.
CPU and CUDA inference are supported. The standard and historical evaluation
protocols are documented in the [inference guide](docs/checkpoints.md#evaluation-protocols).

## Training code

The original training entry point is `train_sat_solar_cast.py`. The original
short-term and long-term launch scripts remain available:

```bash
bash train_bash_sat_solar_cast_short_False1.sh
bash train_bash_sat_solar_cast_long_False1.sh
```

Update their dataset paths and GPU settings for your machine. These scripts
perform training; use `infer.py` to load the released weights. The original
`evaluate.py` computes metrics and plots from saved prediction arrays.

## Validation

```bash
python -m unittest discover -s tests -v
```

The four released weights have also been checked for strict loading and
real-data sample inference on CPU. Full historical paper metrics have not been
revalidated; see the evaluation notes before reporting reproduction results.
