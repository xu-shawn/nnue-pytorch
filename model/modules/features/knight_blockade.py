from .pp_3wide import PP3Wide


class KnightBlockade(PP3Wide):
    HASH = 0x4B424B32
    FEATURE_NAME = "Knight_Blockade"
    INPUT_FEATURE_NAME = "Knight_Blockade"
    MAX_ACTIVE_FEATURES = 8

    NUM_INPUTS = 96
    NUM_REAL_FEATURES = NUM_INPUTS
