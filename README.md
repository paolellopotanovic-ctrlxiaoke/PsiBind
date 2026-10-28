# GSbind

GSbind predicts protein-binding interface probabilities from a single protein
chain. This release contains the pretrained E11 model, structure preprocessing,
and optional extraction of 128-dimensional residue representations.

## Install

Use Python 3.10–3.12 in a fresh environment. The tested environment uses Python
3.12 and PyTorch 2.5.1. For a CUDA machine, install the appropriate PyTorch build
first; CPU inference is the default.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/inference-tested.txt
python -m pip install . --no-deps
```

## Predict

From the downloaded repository directory:

```bash
gsbind predict \
  --structure examples/1ubq.pdb \
  --chain A \
  --checkpoint checkpoints/gsbind_e11.pt \
  --output outputs/1ubq \
  --save-hidden
```

Replace the structure path and chain identifier to predict another protein.
PDB and mmCIF inputs are accepted. The first coordinate model is used. Each run
processes one protein chain; specify `--chain` when several protein chains are
present. Water, non-polymer partners, hydrogen and deuterium are excluded.
For alternate atom locations, the first encountered alternative is retained.
Missing residues are not reconstructed. The output directory must be new or empty.

Use `--device cuda` or `--device cuda:0` for CUDA. `--threads 2` controls CPU
threads. Large proteins require more memory because atomic neighborhoods use
pairwise distances. CPU and CUDA outputs may differ slightly numerically.

## Outputs

| File | Contents |
|---|---|
| `predictions.csv` | One row per retained residue, with chain, original residue number, insertion code, residue name and interface probability |
| `hidden.npz` | Optional `hidden` array of shape `[residues, 128]` and matching `residue_index` |
| `run.json` | Input and checkpoint checksums, preprocessing counts and software versions |

Rows in the CSV and hidden array use the same order. Interface probabilities
refer to protein binding. The raw hidden states are the final fused model
representations; they are not centered or standardized for a downstream probe.

## Example and scope

The included example is ubiquitin, PDB
[1UBQ](https://www.rcsb.org/structure/1UBQ), chain A. Reference CPU predictions
are in `examples/reference/`. The example demonstrates execution and residue
mapping; it is not an accuracy benchmark.

This is an inference release. It includes no training pipeline, dataset rebuild,
checkpoint search, or fitted biological readout heads. The model predicts from
the supplied chain alone; other chains are not included as structural context.
See [MODEL_CARD.md](MODEL_CARD.md) for model scope and preprocessing settings.

```bash
python -m pip install '.[test]'
python -m pytest -q
```

## Code layout

```text
src/gsbind/
├── model/
│   ├── network.py       # Complete GSbind architecture and forward pass
│   ├── config.py        # Encoder architecture configuration
│   └── components/      # Atom encoder, Gaussian projection, GVP and readout
├── preprocessing/      # Structure cleaning, input features and surface generation
├── inference.py        # Checkpoint loading, inference and output
└── cli.py              # gsbind predict command
```

The atom encoder is one component of the complete model. Preprocessing prepares
model inputs and does not contain learned model layers.

## License and attribution

The existing CC BY-NC-SA 4.0 license is retained. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party sources.
