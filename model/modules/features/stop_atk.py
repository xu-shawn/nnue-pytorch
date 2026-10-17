from .pp_3wide import PP3Wide


class StopAtk(PP3Wide):
    """Attackers of the empty square in front of every pawn (its stop square).

    Like Promo_Atk, but for all pawns. Attacker kinds relative to the pawn: own knight,
    bishop, rook, queen, king, then the same five for the enemy; two thermometer levels
    (>= 1, >= 2). Pawn attackers are covered by PP_3Wide and attacks on an occupied stop
    square by the threat features. Index: ((pawn color * 48 + pawn square - 8) * 10 + kind)
    * 2 + level.
    """

    HASH = 0x7D29A4E6
    FEATURE_NAME = "Stop_Atk"
    INPUT_FEATURE_NAME = "Stop_Atk"
    # At most 20 features per pawn and 16 pawns
    MAX_ACTIVE_FEATURES = 16 * 20

    NUM_INPUTS = 2 * 48 * 10 * 2
    NUM_REAL_FEATURES = NUM_INPUTS
