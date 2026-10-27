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
    """Every ram with every other pawn (either colour) on the ram's file or an adjacent file: ram id * 96 +
    (pawn colour != perspective) * 48 + oriented square - 8 (see the loader)."""

    HASH = 0x3A9E1C11
    FEATURE_NAME = "Ram_Pawn_Near"
    INPUT_FEATURE_NAME = "Ram_Pawn_Near"
    MAX_ACTIVE_FEATURES = 32

    NUM_INPUTS = 40 * 96
    NUM_REAL_FEATURES = NUM_INPUTS
