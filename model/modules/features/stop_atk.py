from .pp_3wide import PP3Wide


class StopAtk(PP3Wide):
    """Attackers of the square in front of every pawn (its stop square).

    Like Promo_Atk, but for all pawns. Attacker kinds relative to the pawn: own pawn,
    knight, bishop, rook, queen, king, then the same six for the enemy; two thermometer
    levels (>= 1, >= 2). Index: ((pawn color * 48 + pawn square - 8) * 12 + kind) * 2 + level.
    """

    HASH = 0x7D29A4E5
    FEATURE_NAME = "Stop_Atk"
    INPUT_FEATURE_NAME = "Stop_Atk"
    # At most 24 features per pawn and 16 pawns
    MAX_ACTIVE_FEATURES = 16 * 24

    NUM_INPUTS = 2 * 48 * 12 * 2
    NUM_REAL_FEATURES = NUM_INPUTS
