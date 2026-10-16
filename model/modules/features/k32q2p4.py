from .k32q2 import K32Q2, InverseKingBuckets


class K32Q2P4(K32Q2):
    """K32Q2 with 4 piece-count buckets as a further bucket dimension.

    bucket = pc_bucket * 64 + king_bucket * 2 + opp_queen with pc_bucket = (pieces - 1) // 8,
    so the features of one piece-count bucket form a block laid out exactly like K32Q2.
    """

    HASH = 0x32B5E2C4
    FEATURE_NAME = "K32Q2P4^"
    INPUT_FEATURE_NAME = "K32Q2P4"

    NUM_PC_BUCKETS = 4
    NUM_BUCKETS = K32Q2.NUM_BUCKETS * NUM_PC_BUCKETS  # 256
    NUM_INPUTS = K32Q2.NUM_PLANES * NUM_BUCKETS  # 196,608

    # Export size uses 11 piece types (704 * 256 = 180,224)
    NUM_REAL_FEATURES = 704 * NUM_BUCKETS  # 180,224

    @classmethod
    def own_king_square(cls, bucket: int) -> int:
        return InverseKingBuckets[(bucket // 2) % cls.NUM_KING_BUCKETS]
