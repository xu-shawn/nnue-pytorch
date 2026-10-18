from .pp_3wide import PP3Wide


class PawnPiece3Wide(PP3Wide):
    """Pawn-piece pairs: each knight, bishop, rook, queen or king on the same or an adjacent
    file as a pawn, indexed by the pawn, the piece kind and the piece's square in the window."""

    HASH = 0x5A11C3D1
    FEATURE_NAME = "PawnPiece_3Wide"
    INPUT_FEATURE_NAME = "PawnPiece_3Wide"
    MAX_ACTIVE_FEATURES = 256

    NUM_INPUTS = 20160
    NUM_REAL_FEATURES = NUM_INPUTS
