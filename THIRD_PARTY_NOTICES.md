# Third-party notices

The atomic encoding implementation derives from PeSTo by Krapp et al.,
"Pesto: parameter-free geometric deep learning for accurate prediction of
protein binding interfaces," Nature Communications (2023).
The upstream project is https://github.com/LBM-EPFL/PeSTo and distributes its code
under CC BY-NC-SA 4.0. The corresponding license is retained in `LICENSE`.
The copied/adapted atomic components are in `src/gaussian_direct/atom_encoding/`.
GSbind adds the Gaussian surface/residue interaction model and inference wrapper.

`examples/1ubq.pdb` is the ubiquitin structure 1UBQ from the Protein Data Bank:
Vijay-Kumar S, Bugg CE, Cook WJ. "Structure of ubiquitin refined at 1.8 Å
resolution." Journal of Molecular Biology (1987), DOI:
https://doi.org/10.1016/0022-2836(87)90679-6.
Structure entry: https://www.rcsb.org/structure/1UBQ.

PyTorch, NumPy, SciPy and Gemmi are installed as dependencies and retain their
respective licenses. They are not vendored in this repository.
