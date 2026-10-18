from .pp_3wide import PP3Wide


class KingZoneThreats(PP3Wide):
    """Threats on each king and the empty squares next to it by knights, bishops, rooks and
    queens of either colour, indexed by target square, attacker kind and attacker square."""

    HASH = 0x5A11C3E2
    FEATURE_NAME = "KingZone_Threats"
    INPUT_FEATURE_NAME = "KingZone_Threats"
    MAX_ACTIVE_FEATURES = 288

    NUM_INPUTS = 12992
    NUM_REAL_FEATURES = NUM_INPUTS
