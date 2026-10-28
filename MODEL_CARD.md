# GSbind E11

**Task.** Residue-level protein interface prediction from one protein chain,
without a binding partner as input. The released output is the protein-interface
channel. The final fused residue representation has 128 dimensions.

**Architecture.** Atomic encoding and residue pooling, Gaussian surface
initialization, joint residue-pair Gaussian projection, surface GVP updates,
patch-token attention, and point/region readout into residue updates. The
released architecture uses three interaction passes and the direct protein head.

**Weights.** E11 is the supplied pretrained model. This package re-exports
the same model tensors with an inference-only configuration. The weight file
contains tensors and primitive metadata, with no optimizer state or training
paths. `checkpoints/SHA256SUMS` identifies the released bytes.

**Training background.** The source engineering records 20,801 PINDER-derived
training records, protein-interface supervision, and 20 training epochs. Its
checkpoint-selection record identifies epoch 11 from epochs 0–19 using the
GPSFun AF2 full375 panel and a family-balanced performance score. This release
does not rerun training or checkpoint selection. A panel used for checkpoint
selection must be treated as selection data when interpreting performance.

**Preprocessing.** Heavy atoms from one protein chain in the first coordinate
model are encoded with the released atomic feature order. The surface is
generated with the existing Shrake–Rupley-style implementation: probe radius
1.4 Å, 64 sphere samples per atom, up to 1,024 selected surface points and the
same geometric descriptors as the compact-cache pathway. Model graph sizes,
Gaussian scales and learned parameters are retained from the source implementation.

**Interpretation.** Scores describe predicted protein-binding propensity for
the input structure. They do not specify a partner, establish biochemical
binding, or directly predict allostery. Hidden representations require a
separately specified and validated readout for downstream biological tasks.

**Input limits.** One chain per run. No missing-atom/residue rebuilding or
assembly inference is performed. Non-polymer ligands and other chains are
excluded. Nonstandard amino-acid residues marked as polymer and recognized by
Gemmi are retained; their element features use the existing encoder vocabulary.
Memory use increases with protein size. The tested CPU environment and reference
example are provided; CUDA operation must be checked on the target hardware.
