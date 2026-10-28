"""Read one protein chain while retaining its original residue identifiers."""

from pathlib import Path

import gemmi
import numpy as np
import torch

from gaussian_direct.atom_encoding.encoding import encode_features, encode_structure, extract_topology
from gaussian_direct.atom_encoding.structure import clean_structure
from gaussian_direct.atom_encoding.surface_cache import build_s1_surface_cache_for_version


def prepare_structure(path, chain=None):
    """Return model tensors, an ordered residue map, and preprocessing metadata.

    The first coordinate model is used. One protein chain is required; an
    explicit chain is required if the structure contains several protein chains.
    Water, non-polymer partners, hydrogen and deuterium are excluded.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Structure does not exist: {path}")
    try:
        structure = gemmi.read_structure(str(path))
    except RuntimeError as error:
        raise ValueError(f"Cannot read structure: {error}") from error
    if not len(structure):
        raise ValueError("The structure contains no coordinate models")
    model = structure[0]
    candidates = [
        c for c in model
        if any(r.het_flag == "A" and gemmi.find_tabulated_residue(r.name).is_amino_acid() for r in c)
    ]
    available = [c.name for c in candidates]
    if chain is None:
        if len(candidates) != 1:
            raise ValueError(f"Select one protein chain with --chain; available chains: {available}")
        selected = candidates[0]
    else:
        matches = [c for c in candidates if c.name == chain]
        if len(matches) != 1:
            raise ValueError(f"Protein chain {chain!r} is not unique or absent; available chains: {available}")
        selected = matches[0]

    values = {k: [] for k in ["xyz", "name", "element", "resname", "resid", "het_flag", "chain_name", "icode"]}
    residue_map = []
    seen_residues = set()
    for residue in selected:
        if residue.het_flag != "A" or not gemmi.find_tabulated_residue(residue.name).is_amino_acid():
            continue
        code = residue.seqid.icode.strip()
        identity = (selected.name, residue.seqid.num, code)
        if identity in seen_residues:
            raise ValueError(f"Duplicate residue identity: {identity}")
        seen_residues.add(identity)
        seen_altloc = set()
        kept = []
        for atom in residue:
            if atom.element.name in {"H", "D"}:
                continue
            if atom.has_altloc():
                if atom.name in seen_altloc:
                    continue
                seen_altloc.add(atom.name)
            kept.append(atom)
        if not kept:
            continue
        residue_map.append({
            "residue_index": len(residue_map), "chain_id": selected.name,
            "residue_number": residue.seqid.num, "insertion_code": code,
            "residue_name": residue.name,
        })
        for atom in kept:
            row = [list(atom.pos), atom.name, atom.element.name, residue.name,
                   residue.seqid.num, "A", f"{selected.name}:0", code]
            for key, value in zip(values, row):
                values[key].append(value)
    if not residue_map or len(values["xyz"]) < 2:
        raise ValueError("The selected chain has insufficient protein atoms")
    arrays = {k: np.asarray(v, dtype=np.float32 if k == "xyz" else np.int32 if k == "resid" else None)
              for k, v in values.items()}
    arrays = clean_structure(arrays)
    coordinates, assignment = encode_structure(arrays)
    if not torch.isfinite(coordinates).all():
        raise ValueError("The structure contains non-finite coordinates")
    if assignment.shape[1] != len(residue_map):
        raise ValueError("Residue mapping does not match encoded residues")
    features = torch.cat(encode_features(arrays), dim=1)
    topology = extract_topology(coordinates, 64)[0] + 1
    if topology.shape[1] < 64:
        topology = torch.cat([topology, topology.new_zeros((len(topology), 64-topology.shape[1]))], dim=1)
    # These settings match the released compact-cache preprocessing contract.
    surface = build_s1_surface_cache_for_version(
        {"X": coordinates, "q": features, "M": assignment.float()},
        version="s1_surface_shrake_v1", num_surface_points=1024,
        graph_k=16, lappe_dim=0, shrake_probe_radius=1.4,
        shrake_sphere_samples=64, shrake_neighbor_method="bruteforce",
    )
    metadata = {
        "chain_id": selected.name, "coordinate_model": 0,
        "atoms": len(coordinates), "residues": len(residue_map),
        "surface_points": int(surface["surface_pos"].shape[0]),
        "surface_version": "s1_surface_shrake_v1",
    }
    return {"X": coordinates, "ids_topk": topology, "q0": features[:, :30],
            "M": assignment.float(), "surface_cache": surface}, residue_map, metadata
