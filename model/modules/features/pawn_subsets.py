from .pawnatk_threats import PawnAtkThreats
from .pawnpiece_3wide import PawnPiece3Wide


class PawnKR3Wide(PawnPiece3Wide):
    """PawnPiece_3Wide restricted to kings and rooks (same index)."""

    HASH = 0x5A11C3F3
    FEATURE_NAME = "PawnKR_3Wide"
    INPUT_FEATURE_NAME = "PawnKR_3Wide"


class PawnContacts(PawnAtkThreats):
    """PawnAtk_Threats restricted to the enemy pawns attacking a square the pawns attack."""

    HASH = 0x5A11C404
    FEATURE_NAME = "Pawn_Contacts"
    INPUT_FEATURE_NAME = "Pawn_Contacts"
    MAX_ACTIVE_FEATURES = 64
