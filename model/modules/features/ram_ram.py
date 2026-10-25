from .pp_3wide import PP3Wide


class RamRam(PP3Wide):
    """Every pair of rams on the board, triangular index over the two ram ids."""

    HASH = 0x3A9E1C03
    FEATURE_NAME = "Ram_Ram"
    INPUT_FEATURE_NAME = "Ram_Ram"
    MAX_ACTIVE_FEATURES = 8 * 7 // 2

    NUM_INPUTS = 40 * 39 // 2
    NUM_REAL_FEATURES = NUM_INPUTS
