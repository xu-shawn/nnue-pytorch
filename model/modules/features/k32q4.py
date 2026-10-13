from .k32q2 import K32Q2, InverseKingBuckets


class K32Q4(K32Q2):
    """K32Q2 with the own queen presence as a further bucket dimension.

    bucket = own_queen * 64 + king_bucket * 2 + opp_queen, so the features of one
    own-queen state form a contiguous block that is laid out exactly like K32Q2.
    """

    HASH = 0x34B5E284
    FEATURE_NAME = "K32Q4^"
    INPUT_FEATURE_NAME = "K32Q4"

    NUM_QUEEN_BUCKETS = 4
    NUM_BUCKETS = K32Q2.NUM_KING_BUCKETS * NUM_QUEEN_BUCKETS  # 128
    NUM_INPUTS = K32Q2.NUM_PLANES * NUM_BUCKETS  # 98,304

    # Export size uses 11 piece types (704 * 128 = 90,112)
    NUM_REAL_FEATURES = 704 * NUM_BUCKETS  # 90,112

    @classmethod
    def own_king_square(cls, bucket: int) -> int:
        return InverseKingBuckets[(bucket // 2) % cls.NUM_KING_BUCKETS]
