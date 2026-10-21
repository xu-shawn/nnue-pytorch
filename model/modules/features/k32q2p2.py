from .k32q2 import K32Q2
from .k32q2p4 import K32Q2P4


class K32Q2P2(K32Q2P4):
    """K32Q2 with 2 piece-count buckets as a further bucket dimension.

    bucket = pc_bucket * 64 + king_bucket * 2 + opp_queen with pc_bucket = pieces > 16 (kings and
    pawns counted), so the features of one piece-count bucket form a block laid out exactly like K32Q2.
    """

    HASH = 0x32B5E2C2
    FEATURE_NAME = "K32Q2P2^"
    INPUT_FEATURE_NAME = "K32Q2P2"

    NUM_PC_BUCKETS = 2
    NUM_BUCKETS = K32Q2.NUM_BUCKETS * NUM_PC_BUCKETS  # 128
    NUM_INPUTS = K32Q2.NUM_PLANES * NUM_BUCKETS  # 98,304

    # Export size uses 11 piece types (704 * 128 = 90,112)
    NUM_REAL_FEATURES = 704 * NUM_BUCKETS  # 90,112
