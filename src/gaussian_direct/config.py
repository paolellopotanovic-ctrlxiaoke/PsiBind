"""Architecture constants for the released E11 checkpoint."""

ATOM_ENCODER_CONFIG = {
    "atom_features": {"N0": 30, "N1": 32},
    "state_updates": [
        *(
            {"Ns": 32, "Nh": 2, "Nk": 3, "nn": 8}
            for _ in range(8)
        ),
        *(
            {"Ns": 32, "Nh": 2, "Nk": 3, "nn": 16}
            for _ in range(8)
        ),
        *(
            {"Ns": 32, "Nh": 2, "Nk": 3, "nn": 32}
            for _ in range(8)
        ),
        *(
            {"Ns": 32, "Nh": 2, "Nk": 3, "nn": 64}
            for _ in range(8)
        ),
    ],
    "residue_pool": {"N0": 32, "N1": 32, "Nh": 4},
    "residue_decoder": {"N0": 32, "N1": 32, "N2": 5},
}
