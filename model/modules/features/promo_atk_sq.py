from .pp_3wide import PP3Wide


class PromoAtkSq(PP3Wide):
    """Like Promo_Atk, but each attacker of a promotion square is a feature of its own.

    Index: pawn color, promotion file, attacker kind (own pawn, knight, bishop, rook, queen,
    king, enemy knight, bishop, rook, queen, king) and attacker square, counting only the
    squares from which that kind can attack the promotion square.
    """

    HASH = 0x7D29A4D1
    FEATURE_NAME = "Promo_AtkSq"
    INPUT_FEATURE_NAME = "Promo_AtkSq"
    # At most 17 attackers per promotion square and 16 pawns on their 7th rank
    MAX_ACTIVE_FEATURES = 16 * 17

    NUM_INPUTS = 1620
    NUM_REAL_FEATURES = NUM_INPUTS
