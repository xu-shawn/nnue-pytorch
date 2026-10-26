from .k32q2 import K32Q2, InverseKingBuckets


class K32H3B(K32Q2):
    """K32H3 (32 own-king buckets x 3 piece-square states) laid out block-major:

    bucket = state * 32 + king_bucket, with the states in order of increasing heavy material
      0: no rook or queen on the board (either side),
      1: some rook or queen on the board, but the opponent of the perspective has no queen,
      2: the opponent of the perspective has a queen.
    (No rook or queen on the board with an opponent queen cannot occur, hence 3 states, not 4.)

    The state is the engine's piece-square bucket, so each state's 22,528 exported rows
    (32 x 704) are one contiguous block that the engine indexes like HalfKAv2_hm. Exported in this
    order (67,584 rows). K32H3 holds the same weights in a legacy interleave;
    scripts/convert_hm_layout.py converts nets.
    """

    HASH = 0x7ACE3084
    FEATURE_NAME = "K32H3B^"
    INPUT_FEATURE_NAME = "K32H3B"

    NUM_STATES = 3
    NUM_BUCKETS = K32Q2.NUM_KING_BUCKETS * NUM_STATES  # 96
    NUM_INPUTS = K32Q2.NUM_PLANES * NUM_BUCKETS  # 73,728

    # Export size uses 11 piece types (704 * 96 = 67,584)
    NUM_REAL_FEATURES = 704 * NUM_BUCKETS  # 67,584

    @classmethod
    def own_king_square(cls, bucket: int) -> int:
        return InverseKingBuckets[bucket % cls.NUM_KING_BUCKETS]
