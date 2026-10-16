from .pp_3wide import PP3Wide


class PawnPush(PP3Wide):
    """Pawns whose push square holds a non-pawn piece of either color.

    Index: (pawn color * 10 + blocker color * 5 + blocker type) * 48 + pawn square,
    colors relative to the perspective, squares oriented like Full_Threats.
    """

    HASH = 0x5C3A91E7
    FEATURE_NAME = "Pawn_Push"
    INPUT_FEATURE_NAME = "Pawn_Push"
    MAX_ACTIVE_FEATURES = 16

    NUM_INPUTS = 2 * 10 * 48
    NUM_REAL_FEATURES = NUM_INPUTS
