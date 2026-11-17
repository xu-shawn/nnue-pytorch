from .pp_3wide import PP3Wide


class KnightPawn(PP3Wide):
    HASH = 0x4B4E5031
    FEATURE_NAME = "Knight_Pawn"
    INPUT_FEATURE_NAME = "Knight_Pawn"
    MAX_ACTIVE_FEATURES = 32

    NUM_INPUTS = 96 * 11
    NUM_REAL_FEATURES = NUM_INPUTS


class KnightBehind(PP3Wide):
    """Slot 0 of Knight_Pawn alone: a knight right behind a pawn of its own side."""

    HASH = 0x4B4E4231
    FEATURE_NAME = "Knight_Behind"
    INPUT_FEATURE_NAME = "Knight_Behind"
    MAX_ACTIVE_FEATURES = 16

    NUM_INPUTS = 96
    NUM_REAL_FEATURES = NUM_INPUTS


class BishopDomination(PP3Wide):
    HASH = 0x42444D31
    FEATURE_NAME = "Bishop_Domination"
    INPUT_FEATURE_NAME = "Bishop_Domination"
    MAX_ACTIVE_FEATURES = 16

    NUM_INPUTS = 2 * 64
    NUM_REAL_FEATURES = NUM_INPUTS
