"""Compact-input tensor adapter; no training dependency."""
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

def split_field(value: object, cast=str) -> list[Any]:
    text = str(value or "")
    return [cast(item) for item in text.split(";") if item != ""]

def _surface(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if not int(payload["has_s1_surface"].item()):
        raise ValueError("Gaussian-direct128 requires an NPZ with has_s1_surface=1")
    surface: dict[str, Any] = {}
    for name in (
        "surface_pos",
        "surface_normal",
        "surface_geom",
        "surface_graph_dist",
        "surface_lappe",
        "nearest_residue_dist",
        "surface_mask",
    ):
        surface[name] = torch.from_numpy(payload[name].copy())
    for name in ("surface_graph_idx", "nearest_residue_idx"):
        surface[name] = torch.from_numpy(payload[name].astype(np.int64)).long()
    if "surface_graph_ks" in payload:
        graph_ks = tuple(int(value) for value in payload["surface_graph_ks"].tolist())
        surface["surface_graph_ks"] = graph_ks
        for graph_k in graph_ks:
            surface[f"surface_graph_idx_k{graph_k}"] = torch.from_numpy(
                payload[f"surface_graph_idx_k{graph_k}"].astype(np.int64)
            ).long()
            surface[f"surface_graph_dist_k{graph_k}"] = torch.from_numpy(
                payload[f"surface_graph_dist_k{graph_k}"].copy()
            )
    surface.update(
        {
            "num_surface_points": int(payload["surface_num_surface_points"].item()),
            "num_residues": int(payload["surface_num_residues"].item()),
            "cache_kind": str(payload["surface_cache_kind"].item()),
            "probe_radius": float(payload["surface_probe_radius"].item()),
            "sphere_samples": int(payload["surface_sphere_samples"].item()),
            "accessible_candidates": int(payload["surface_accessible_candidates"].item()),
        }
    )
    for name in ("source_atom_idx", "surface_element_idx"):
        if name in payload:
            surface[name] = torch.from_numpy(payload[name].astype(np.int64)).long()
    for name in ("surface_chemistry", "surface_curvature", "residue_sasa"):
        if name in payload:
            surface[name] = torch.from_numpy(payload[name].copy())
    for name in ("surface_graph_mode", "surface_point_mode", "semantic_schema"):
        if name in payload:
            surface[name] = str(payload[name].item())
    for name in ("surface_point_density", "surface_area"):
        if name in payload:
            surface[name] = float(payload[name].item())
    if "surface_point_max" in payload:
        surface["surface_point_max"] = int(payload["surface_point_max"].item())
    return surface

def load_compact_npz(
    path: Path | str,
    *,
    case_id: str = "",
    labels: object = "",
    label_mask: object = "",
    input_state: str = "holo",
) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        num_residues = int(payload["num_residues"].item())
        atom_to_residue = torch.from_numpy(payload["atom_to_residue_index"].astype(np.int64)).long()
        if atom_to_residue.numel() and int(atom_to_residue.max()) >= num_residues:
            raise ValueError(f"Invalid atom-to-residue index in {path}")
        assignment = torch.nn.functional.one_hot(atom_to_residue, num_classes=num_residues).float()
        zero_based_ids = torch.from_numpy(payload["ids_topk"].astype(np.int64)).long()
        if zero_based_ids.numel() and (zero_based_ids.min() < 0 or zero_based_ids.max() >= len(payload["X"])):
            raise ValueError(f"Invalid zero-based atom topology in {path}")
        ids = zero_based_ids + 1
        if ids.shape[1] < 64:
            ids = torch.cat([ids, ids.new_zeros((ids.shape[0], 64 - ids.shape[1]))], dim=1)
        item = {
            "X": torch.from_numpy(payload["X"].copy()).float(),
            "ids_topk": ids[:, :64],
            "q": torch.from_numpy(payload["q"][:, :30].copy()).float(),
            "M": assignment,
            "surface_cache": _surface(payload),
            "case_id": case_id or path.stem,
            "input_state": input_state,
        }
    labels_tensor = torch.tensor(split_field(labels, int), dtype=torch.float32)
    mask_tensor = torch.tensor(split_field(label_mask, int), dtype=torch.float32)
    if labels_tensor.numel() == 0:
        labels_tensor = torch.zeros(num_residues)
    if mask_tensor.numel() == 0:
        mask_tensor = torch.ones(num_residues)
    if labels_tensor.numel() != num_residues or mask_tensor.numel() != num_residues:
        raise ValueError(f"Label length mismatch for {path}: expected {num_residues}")
    item.update({"y": labels_tensor, "mask": mask_tensor})
    return item

def collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Batches must be non-empty")
    if len(batch) == 1:
        item = batch[0]
        return {
            **item,
            "surface_cache": [item["surface_cache"]],
            "row": [item.get("row", {"apex_case_id": item["case_id"]})],
            "atom_counts": [int(item["X"].shape[0])],
            "residue_counts": [int(item["M"].shape[1])],
        }
    atom_counts = [int(item["X"].shape[0]) for item in batch]
    residue_counts = [int(item["M"].shape[1]) for item in batch]
    total_atoms = sum(atom_counts)
    X = torch.cat([item["X"] for item in batch], dim=0)
    q = torch.cat([item["q"] for item in batch], dim=0)
    ids_topk = X.new_zeros((total_atoms, 64), dtype=torch.long)
    atom_start = residue_start = 0
    for item, atom_count in zip(batch, atom_counts):
        atom_end = atom_start + atom_count
        local_ids = item["ids_topk"]
        valid = local_ids > 0
        global_ids = local_ids.clone()
        global_ids[valid] += atom_start
        ids_topk[atom_start:atom_end] = global_ids
        atom_start = atom_end
        residue_start += int(item["M"].shape[1])
    return {
        "X": X,
        "ids_topk": ids_topk,
        "q": q,
        "M": [item["M"] for item in batch],
        "y": torch.cat([item["y"] for item in batch]),
        "mask": torch.cat([item["mask"] for item in batch]),
        "surface_cache": [item["surface_cache"] for item in batch],
        "row": [item.get("row", {"apex_case_id": item["case_id"]}) for item in batch],
        "atom_counts": atom_counts,
        "residue_counts": residue_counts,
    }

def model_arguments_from_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    X = batch["X"].to(device, non_blocking=True)
    M_parts = batch["M"]
    if isinstance(M_parts, torch.Tensor):
        M = M_parts.to(device, non_blocking=True)
    else:
        atom_counts = [int(part.shape[0]) for part in M_parts]
        residue_counts = [int(part.shape[1]) for part in M_parts]
        M = M_parts[0].new_zeros((sum(atom_counts), sum(residue_counts)), device=device)
        atom_start = residue_start = 0
        for part, atom_count, residue_count in zip(M_parts, atom_counts, residue_counts):
            M[atom_start:atom_start + atom_count, residue_start:residue_start + residue_count] = part.to(device)
            atom_start += atom_count
            residue_start += residue_count
    surface = []
    for current in batch["surface_cache"]:
        surface.append(
            {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in current.items()
            }
        )
    return {
        "X": X,
        "ids_topk": batch["ids_topk"].to(device, non_blocking=True),
        "q0": batch["q"].to(device, non_blocking=True),
        "M": M,
        "residue_centers": None,
        "surface_cache": surface,
        "atom_counts": batch["atom_counts"],
        "residue_counts": batch["residue_counts"],
        "frozen_residue_q": None,
        "frozen_residue_p": None,
    }
