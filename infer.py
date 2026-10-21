#!/usr/bin/env python3
"""Run pretrained CrossViViT or LSTM-att+Sat without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from models import build_model

REPO_ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def predict_batch(model, x, model_name: str, protocol: str):
    """x is normalized BTCHW history; future targets are never passed in."""
    if model_name == "crossvivit":
        return model(x[:, :, :4], x[:, :, 4:5, 0, 0], mask=protocol == "legacy")[
            0
        ].squeeze(-1)
    return model(x, phase=2)


def summarize_metrics(prediction, target):
    error = prediction.astype(np.float64).ravel() - target.astype(np.float64).ravel()
    p, t = prediction.ravel(), target.ravel()
    correlation_squared = None
    if len(p) > 1 and np.std(p) > 0 and np.std(t) > 0:
        correlation_squared = float(np.corrcoef(p, t)[0, 1] ** 2)
    return dict(
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        bias=float(np.mean(error)),
        pearson_r_squared=correlation_squared,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="Override the weight path in the config."
    )
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=["standard", "legacy"],
        default="standard",
        help="standard: eval/no masking; legacy: historical masking/BatchNorm behavior.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count.")
    parser.add_argument(
        "--limit", type=int, help="Process only the first N samples for a quick check."
    )
    parser.add_argument("--units", choices=["wm2", "normalized"], default="wm2")
    args = parser.parse_args()
    for key in ["batch_size", "threads", "limit"]:
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    checkpoint = args.checkpoint or REPO_ROOT / config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{checkpoint}: download and extract the checkpoint archive first; see docs/checkpoints.md"
        )
    checksum = file_sha256(checkpoint)
    if checksum != config["sha256"]:
        raise ValueError(
            f"Checkpoint SHA-256 does not match {config['name']}: {checksum}"
        )
    if any(
        args.output_dir.joinpath(name).exists()
        for name in ["predictions.npy", "targets.npy", "run.json"]
    ):
        raise FileExistsError(
            "Use a new output directory to preserve previous predictions."
        )
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    batch_size = args.batch_size or (
        32 if args.protocol == "legacy" and config["model"] == "crossvivit" else 1
    )
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = build_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    if args.protocol == "legacy" and config["model"] == "lstm-att-sat":
        model.train()  # Historical test code used batch statistics in BatchNorm.
    # Seed after construction so the run's masking sequence is explicit.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with h5py.File(args.input_h5, "r") as data:
        x_data = data["arr_0"]
        expected = (config["input_len"], 120, 120, 5)
        if x_data.ndim != 5 or x_data.shape[1:] != expected:
            raise ValueError(
                f"arr_0 must have shape (N,{expected}), got {x_data.shape}"
            )
        count = min(len(x_data), args.limit) if args.limit else len(x_data)
        if count == 0:
            raise ValueError("Input contains no samples.")
        y_data = data.get("arr_1")
        if y_data is not None and y_data.shape != (
            len(x_data),
            config["output_len"],
            120,
            120,
            5,
        ):
            raise ValueError(f"arr_1 has an unexpected shape: {y_data.shape}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions = np.lib.format.open_memmap(
            args.output_dir / "predictions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, config["output_len"], 1),
        )
        targets = None
        if y_data is not None:
            targets = np.lib.format.open_memmap(
                args.output_dir / "targets.npy",
                mode="w+",
                dtype=np.float32,
                shape=predictions.shape,
            )
        with torch.inference_mode():
            for start in range(0, count, batch_size):
                end = min(start + batch_size, count)
                array = np.asarray(x_data[start:end], dtype=np.float32)
                if not np.isfinite(array).all():
                    raise ValueError(
                        "arr_0 contains non-finite values; run preprocessing first."
                    )
                x = (
                    torch.from_numpy(array)
                    .permute(0, 1, 4, 2, 3)
                    .contiguous()
                    .to(device)
                )
                prediction = (
                    predict_batch(model, x, config["model"], args.protocol)
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(prediction).all():
                    raise ValueError("Model produced non-finite predictions.")
                if args.units == "wm2":
                    prediction = prediction * 279.49 + 193.44
                predictions[start:end] = prediction
                if targets is not None:
                    target = y_data[start:end, :, 0, 0, 4:5]
                    if not np.isfinite(target).all():
                        raise ValueError(
                            "arr_1 in-situ targets contain non-finite values."
                        )
                    targets[start:end] = (
                        target * 279.49 + 193.44 if args.units == "wm2" else target
                    )
        predictions.flush()
        if targets is not None:
            targets.flush()
        result = dict(
            config=config,
            checkpoint_sha256=checksum,
            input_h5=str(args.input_h5),
            samples=count,
            units=args.units,
            protocol=args.protocol,
            batch_size=batch_size,
            seed=args.seed,
            device=device,
            torch_version=torch.__version__,
        )
        if targets is not None:
            result["metrics"] = summarize_metrics(predictions, targets)
        (args.output_dir / "run.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
