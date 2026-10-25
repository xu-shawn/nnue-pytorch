from .pp_3wide import PP3Wide


class XRayThreatsN4(PP3Wide):
    """Slider x-rays past exactly one piece while at most four knights, bishops, rooks and queens are
    on the board: attacker kind and square, target kind and square, blocker colour vs the attacker's
    (see the loader). 99456 inputs, at most 32 real (four sliders, eight lines each)."""

    HASH = 0x58A40004
    FEATURE_NAME = "XRay_Threats_N4"
    INPUT_FEATURE_NAME = "XRay_Threats_N4"
    MAX_ACTIVE_FEATURES = 56

    NUM_INPUTS = 2 * (364 + 672 + 1036) * 24
    NUM_REAL_FEATURES = NUM_INPUTS


class AttackedStateN8(PP3Wide):
    """Every attacked non-king piece with its attacker / defender count state, while at most eight
    knights, bishops, rooks and queens are on the board (see the loader). 3840 inputs."""

    HASH = 0xA77AC408
    FEATURE_NAME = "Attacked_State_N8"
    INPUT_FEATURE_NAME = "Attacked_State_N8"
    MAX_ACTIVE_FEATURES = 56

    NUM_INPUTS = 10 * 64 * 6
    NUM_REAL_FEATURES = NUM_INPUTS


class AttackedStateN10(AttackedStateN8):
    """Attacked_State with at most ten knights, bishops, rooks and queens on the board."""

    HASH = 0xA77AC40A
    FEATURE_NAME = "Attacked_State_N10"
    INPUT_FEATURE_NAME = "Attacked_State_N10"
