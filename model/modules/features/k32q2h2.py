from .k32q2p2 import K32Q2P2


class K32Q2H2(K32Q2P2):
    """K32Q2 with 2 buckets by heavy pieces as a further bucket dimension.

    bucket = heavy * 64 + king_bucket * 2 + opp_queen with heavy = 1 when any rook or queen of
    either side is on the board, so each bucket's features form a block laid out exactly like K32Q2
    (same sizes as K32Q2P2).
    """

    HASH = 0x32B5E28A
    FEATURE_NAME = "K32Q2H2^"
    INPUT_FEATURE_NAME = "K32Q2H2"
