from .k32q2 import K32Q2, InverseKingBuckets


class K32H3(K32Q2):
    """Compact layout of K32Q2H2 (piece-square buckets by heavy pieces): the (no rook or queen on the
    board, opponent queen) combination cannot occur, so each king bucket has 3 piece-square states.

    bucket = king_bucket * 2 + opp_queen (0..63) while any rook or queen is on the board, laid out
    exactly like K32Q2, else 64 + king_bucket. Exported in this order, it is the engine's compact
    layout (45,056 + 22,528 rows).
    """

    HASH = 0x32B5E2E3
    FEATURE_NAME = "K32H3^"
    INPUT_FEATURE_NAME = "K32H3"

    NUM_BUCKETS = K32Q2.NUM_KING_BUCKETS * 3  # 96
    NUM_INPUTS = K32Q2.NUM_PLANES * NUM_BUCKETS  # 73,728

    # Export size uses 11 piece types (704 * 96 = 67,584)
    NUM_REAL_FEATURES = 704 * NUM_BUCKETS  # 67,584

    @classmethod
    def own_king_square(cls, bucket: int) -> int:
        kb = bucket // 2 if bucket < 2 * cls.NUM_KING_BUCKETS else bucket - 2 * cls.NUM_KING_BUCKETS
        return InverseKingBuckets[kb]
