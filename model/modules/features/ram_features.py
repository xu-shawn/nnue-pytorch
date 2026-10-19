from .passstop_threats import PassStopThreats
from .pp_3wide import PP3Wide


class PassStopThreatsN4(PassStopThreats):
    """PassStop_Threats only with at most 4 knights, bishops, rooks and queens on the board."""

    HASH = 0x7D29A4F8
    FEATURE_NAME = "PassStop_Threats_N4"
    INPUT_FEATURE_NAME = "PassStop_Threats_N4"


class PassStopThreatsN8(PassStopThreats):
    """PassStop_Threats only with at most 8 knights, bishops, rooks and queens on the board."""

    HASH = 0x7D29A4F9
    FEATURE_NAME = "PassStop_Threats_N8"
    INPUT_FEATURE_NAME = "PassStop_Threats_N8"


class PassStopThreatsN12(PassStopThreats):
    """PassStop_Threats only with at most 12 knights, bishops, rooks and queens on the board."""

    HASH = 0x7D29A4FA
    FEATURE_NAME = "PassStop_Threats_N12"
    INPUT_FEATURE_NAME = "PassStop_Threats_N12"


class RamPawn(PP3Wide):
    """Every ram (a pawn blocked head-on by an enemy pawn) paired with every other pawn.

    Index: ram id * 96 + other pawn (color relative to the perspective * 48 + oriented
    square - 8). The ram id is the oriented square of the ram's lower pawn, ranks 2 to 6.
    """

    HASH = 0x3A9E1C01
    FEATURE_NAME = "Ram_Pawn"
    INPUT_FEATURE_NAME = "Ram_Pawn"
    MAX_ACTIVE_FEATURES = 8 * 14

    NUM_INPUTS = 40 * 96
    NUM_REAL_FEATURES = NUM_INPUTS


class RamPawnFile(PP3Wide):
    """Every ram paired with the files that hold another pawn of each color.

    Index: ram id * 16 + color relative to the perspective * 8 + oriented file.
    """

    HASH = 0x3A9E1C02
    FEATURE_NAME = "Ram_PawnFile"
    INPUT_FEATURE_NAME = "Ram_PawnFile"
    MAX_ACTIVE_FEATURES = 8 * 16

    NUM_INPUTS = 40 * 16
    NUM_REAL_FEATURES = NUM_INPUTS


class RamRam(PP3Wide):
    """Every pair of rams on the board, triangular index over the two ram ids."""

    HASH = 0x3A9E1C03
    FEATURE_NAME = "Ram_Ram"
    INPUT_FEATURE_NAME = "Ram_Ram"
    MAX_ACTIVE_FEATURES = 8 * 7 // 2

    NUM_INPUTS = 40 * 39 // 2
    NUM_REAL_FEATURES = NUM_INPUTS
