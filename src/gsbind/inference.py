"""Checkpoint loading and label-free inference."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gaussian_direct.config import ATOM_ENCODER_CONFIG
from gaussian_direct.model import GaussianDirectModel
from .structure import prepare_structure


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint, device="cpu"):
    """Load the release's tensor-only checkpoint with strict parameter matching."""
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "gsbind_inference_v1":
        raise ValueError("Expected a GSbind inference release checkpoint")
    config = payload["model_config"]
    model = GaussianDirectModel(encoder_config=ATOM_ENCODER_CONFIG, **config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval().to(device)
    return model


def forward(model, tensors, device="cpu"):
    """Return interface probabilities and final fused hidden states, in order."""
    device = torch.device(device)
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value
    with torch.inference_mode():
        logits = model(**move(tensors))
        probability = logits[:, 0].sigmoid().float().cpu().numpy()
        hidden = model.gaussian.last_fused.detach().float().cpu().numpy()
    if hidden.shape != (len(probability), 128) or not np.isfinite(hidden).all() or not np.isfinite(probability).all():
        raise RuntimeError("Model returned invalid scores or hidden states")
    return probability, hidden


def predict(structure, checkpoint, output, chain=None, device="cpu", save_hidden=False, threads=2):
    """Predict one chain and write aligned CSV, optional NPZ, and run metadata."""
    if threads < 1:
        raise ValueError("threads must be positive")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}")
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    np.random.seed(42)
    model = load_model(checkpoint, device)
    tensors, residue_map, metadata = prepare_structure(structure, chain)
    probability, hidden = forward(model, tensors, device)
    output.mkdir(parents=True, exist_ok=True)
    rows = [{**row, "interface_probability": float(score)} for row, score in zip(residue_map, probability)]
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if save_hidden:
        np.savez_compressed(output / "hidden.npz", hidden=hidden,
                            residue_index=np.arange(len(rows), dtype=np.int64))
    metadata.update({"model": "GSbind E11", "release": "0.1.0", "device": str(device),
                     "checkpoint_sha256": sha256(checkpoint), "input_sha256": sha256(structure),
                     "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                     "hidden_saved": bool(save_hidden)})
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output / "predictions.csv"
