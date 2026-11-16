from .pp_3wide import PP3Wide


class BishopTriangle(PP3Wide):
    HASH = 0x42545231
    FEATURE_NAME = "Bishop_Triangle"
    INPUT_FEATURE_NAME = "Bishop_Triangle"
    MAX_ACTIVE_FEATURES = 10

    NUM_INPUTS = 2 * 64 * 16
    NUM_REAL_FEATURES = NUM_INPUTS
