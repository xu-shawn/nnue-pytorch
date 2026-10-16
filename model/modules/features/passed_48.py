from .pp_3wide import PP3Wide


class Passed48(PP3Wide):
    """The perspective's own passed pawns, indexed by oriented square (ranks 2 to 7)."""

    HASH = 0x3B8E51F6
    FEATURE_NAME = "Passed_48"
    INPUT_FEATURE_NAME = "Passed_48"
    MAX_ACTIVE_FEATURES = 16

    NUM_INPUTS = 48
    NUM_REAL_FEATURES = NUM_INPUTS
