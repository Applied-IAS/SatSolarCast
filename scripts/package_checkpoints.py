#!/usr/bin/env python3
"""Build the weights-only Google Drive archive from the original experiment tree."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "crossvivit_3h": "crossvivit/model_shortFalse0/best_model.pth",
    "crossvivit_6h": "crossvivit/model_24False0/best_model.pth",
    "lstm-att-sat_3h": "lstm-att/model_shortFalse1/best_model.pth",
    "lstm-att-sat_6h": "lstm-att/model_24False1/best_model.pth",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text())
    for entry in manifest["checkpoints"]:
        source = args.source_root / SOURCES[entry["name"]]
        with source.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        if checksum != entry["sha256"] or source.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Source checkpoint does not match manifest: {source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "x", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for entry in manifest["checkpoints"]:
            archive.write(
                args.source_root / SOURCES[entry["name"]], entry["checkpoint"]
            )
        # Keep the Drive URL in Git; extracting weights must not overwrite it
        # with the upload-pending metadata used when this archive was built.
        archive.writestr(
            "checkpoints/weights-manifest.json",
            json.dumps({"checkpoints": manifest["checkpoints"]}, indent=2) + "\n",
        )
        archive.write(ROOT / "checkpoints/SHA256SUMS", "checkpoints/SHA256SUMS")
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
