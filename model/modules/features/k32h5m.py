from .k32q2 import K32Q2, InverseKingBuckets


class K32H5M(K32Q2):
    """K32H3 with its two heavy states split by whether any knight or bishop is on the board
    (5 piece-square states per king bucket).

    bucket = king_bucket * 2 + opp_queen (0..63) with some rook or queen and some minor on the board,
    64 + king_bucket with no rook or queen, 96 + king_bucket * 2 + opp_queen with some rook or queen
    and no minor. Proxy layout: the no-minor states keep their (never active) knight and bishop rows.
    """

    HASH = 0x32B5E2E5
    FEATURE_NAME = "K32H5M^"
    INPUT_FEATURE_NAME = "K32H5M"

    NUM_BUCKETS = K32Q2.NUM_KING_BUCKETS * 5  # 160
    NUM_INPUTS = K32Q2.NUM_PLANES * NUM_BUCKETS  # 122,880

    NUM_REAL_FEATURES = 704 * NUM_BUCKETS  # 112,640

    @classmethod
    def own_king_square(cls, bucket: int) -> int:
        n = cls.NUM_KING_BUCKETS
        if bucket < 2 * n:
            kb = bucket // 2
        elif bucket < 3 * n:
            kb = bucket - 2 * n
        else:
            kb = (bucket - 3 * n) // 2
        return InverseKingBuckets[kb]
