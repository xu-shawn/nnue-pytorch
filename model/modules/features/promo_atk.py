from .pp_3wide import PP3Wide


class PromoAtk(PP3Wide):
    """Attackers of the promotion square of each pawn on its 7th rank.

    Attacker kinds relative to the pawn: own pawn, knight, bishop, rook, queen, king,
    enemy knight, bishop, rook, queen, king; two thermometer levels (>= 1, >= 2).
    Index: ((pawn color * 8 + promotion file) * 11 + kind) * 2 + level.
    """

    HASH = 0x7D29A4C3
    FEATURE_NAME = "Promo_Atk"
    INPUT_FEATURE_NAME = "Promo_Atk"
    # At most 20 features per pawn and 16 pawns on their 7th rank
    MAX_ACTIVE_FEATURES = 16 * 20

    NUM_INPUTS = 2 * 8 * 11 * 2
    NUM_REAL_FEATURES = NUM_INPUTS
