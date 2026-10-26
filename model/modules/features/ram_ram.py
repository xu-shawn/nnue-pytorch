from .pp_3wide import PP3Wide


class RamRam(PP3Wide):
    """Every pair of rams on the board, triangular index over the two ram ids."""

    HASH = 0x3A9E1C03
    FEATURE_NAME = "Ram_Ram"
    INPUT_FEATURE_NAME = "Ram_Ram"
    MAX_ACTIVE_FEATURES = 8 * 7 // 2

    NUM_INPUTS = 40 * 39 // 2
    NUM_REAL_FEATURES = NUM_INPUTS


class RamRamHm(RamRam):
    """Ram_Ram oriented like HalfKAv2_hm (mirrored when the king is on files a..d, see
    FullThreatsHm). Same index space; each ram id becomes id ^ 7 (its file mirrored)."""

    HASH = 0x5E9FCAC8
    FEATURE_NAME = "Ram_Ram_hm"
    INPUT_FEATURE_NAME = "Ram_Ram_hm"
