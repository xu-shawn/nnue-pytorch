from .pp_3wide import PP3Wide


class KnightMobCover(PP3Wide):
    HASH = 0x4B4D4332
    FEATURE_NAME = "Knight_MobCover"
    INPUT_FEATURE_NAME = "Knight_MobCover"
    MAX_ACTIVE_FEATURES = 64

    NUM_INPUTS = 2 * 64 * 48
    NUM_REAL_FEATURES = NUM_INPUTS
