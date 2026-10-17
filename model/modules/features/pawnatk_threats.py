from .pp_3wide import PP3Wide


class PawnAtkThreats(PP3Wide):
    """Threat features for the empty squares attacked by pawns.

    For each color, every empty square its pawns attack is a target, and every knight,
    bishop, rook, queen or king attacking it is a feature of its own, indexed by attacker
    square (kinds relative to the pawns' color). Two pawn kinds add what PP_3Wide misses:
    the target is attacked by two of the color's pawns, and each enemy pawn attacking it.
    """

    HASH = 0x7D29A501
    FEATURE_NAME = "PawnAtk_Threats"
    INPUT_FEATURE_NAME = "PawnAtk_Threats"
    # At most 32 targets with 21 features each
    MAX_ACTIVE_FEATURES = 32 * 21

    NUM_INPUTS = 11464
    NUM_REAL_FEATURES = NUM_INPUTS
