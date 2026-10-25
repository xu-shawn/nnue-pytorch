from .pp_3wide import PP3Wide


class RamRam(PP3Wide):
    """Every pair of rams on the board, triangular index over the two ram ids."""

    HASH = 0x3A9E1C03
    FEATURE_NAME = "Ram_Ram"
    INPUT_FEATURE_NAME = "Ram_Ram"
    MAX_ACTIVE_FEATURES = 8 * 7 // 2

    NUM_INPUTS = 40 * 39 // 2
    NUM_REAL_FEATURES = NUM_INPUTS


class RamPawnNear(PP3Wide):
    """Every ram with every other pawn on the ram's file or an adjacent file; index as Ram_Pawn."""

    HASH = 0x3A9E1C11
    FEATURE_NAME = "Ram_Pawn_Near"
    INPUT_FEATURE_NAME = "Ram_Pawn_Near"
    MAX_ACTIVE_FEATURES = 24

    NUM_INPUTS = 40 * 96
    NUM_REAL_FEATURES = NUM_INPUTS


class RamSupport(PP3Wide):
    """Every ram with every same-colour pawn diagonally adjacent to one of its pawns (see the loader)."""

    HASH = 0x3A9E1C21
    FEATURE_NAME = "Ram_Support"
    INPUT_FEATURE_NAME = "Ram_Support"
    MAX_ACTIVE_FEATURES = 24

    NUM_INPUTS = 40 * 8
    NUM_REAL_FEATURES = NUM_INPUTS


class Pad24(PP3Wide):
    """Reserves 24 slots and is never active (one row): keeps a base run on the 288-slot kernels."""

    HASH = 0x9AD00024
    FEATURE_NAME = "Pad24"
    INPUT_FEATURE_NAME = "Pad24"
    MAX_ACTIVE_FEATURES = 24

    NUM_INPUTS = 1
    NUM_REAL_FEATURES = NUM_INPUTS
