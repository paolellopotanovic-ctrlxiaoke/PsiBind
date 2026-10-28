import csv
from pathlib import Path
import subprocess
import sys

import gemmi
import numpy as np
import pytest
import torch

from gsbind.inference import load_model, sha256
from gsbind.preprocessing.structure import prepare_structure

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_and_dependency_boundary():
    model = load_model(ROOT / "checkpoints/gsbind_e11.pt")
    assert not model.training
    assert "lightning" not in sys.modules
    expected = (ROOT / "checkpoints/SHA256SUMS").read_text().split()[0]
    assert sha256(ROOT / "checkpoints/gsbind_e11.pt") == expected


def test_cli_full_protein(tmp_path):
    output = tmp_path / "predictions"
    subprocess.run([
        sys.executable, "-m", "gsbind", "predict",
        "--structure", str(ROOT / "examples/1ubq.pdb"), "--chain", "A",
        "--checkpoint", str(ROOT / "checkpoints/gsbind_e11.pt"),
        "--output", str(output), "--save-hidden",
    ], cwd=tmp_path, check=True, capture_output=True, text=True)
    with (output / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "examples/reference/predictions.csv").open() as stream:
        reference = list(csv.DictReader(stream))
    assert len(rows) == len(reference) == 76
    for current, expected in zip(rows, reference):
        for key in ["chain_id", "residue_number", "insertion_code", "residue_name", "residue_index"]:
            assert current[key] == expected[key]
    np.testing.assert_allclose(
        [float(r["interface_probability"]) for r in rows],
        [float(r["interface_probability"]) for r in reference], rtol=1e-4, atol=1e-5,
    )
    with np.load(output / "hidden.npz", allow_pickle=False) as arrays:
        assert arrays["hidden"].shape == (76, 128)
        assert np.isfinite(arrays["hidden"]).all()
        np.testing.assert_array_equal(arrays["residue_index"], np.arange(76))


def test_structure_mapping_and_mmcif(tmp_path, monkeypatch):
    # Isolate identifier handling; the CLI test exercises the real surface builder.
    monkeypatch.setattr("gsbind.preprocessing.structure.build_s1_surface_cache_for_version",
                        lambda item, **kwargs: {"surface_pos": torch.zeros(1, 3)})
    structure = gemmi.read_structure(str(ROOT / "examples/1ubq.pdb"))
    chain = structure[0][0]
    chain[0].seqid = gemmi.SeqId(-2, " ")
    chain[1].seqid = gemmi.SeqId(10, "A")
    chain[2].seqid = gemmi.SeqId(10, "B")
    extra = chain.clone()
    extra.name = "B"
    structure[0].add_chain(extra)
    path = tmp_path / "identifiers.cif"
    structure.make_mmcif_document().write_file(str(path))
    with pytest.raises(ValueError, match="Select one protein chain"):
        prepare_structure(path)
    tensors, rows, metadata = prepare_structure(path, "A")
    assert [(r["residue_number"], r["insertion_code"]) for r in rows[:3]] == [(-2, ""), (10, "A"), (10, "B")]
    assert tensors["M"].shape[1] == 76
    assert metadata["chain_id"] == "A"
    with pytest.raises(ValueError, match="absent"):
        prepare_structure(path, "Z")
