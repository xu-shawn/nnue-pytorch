from .pp_3wide import PP3Wide


class PPWide2(PP3Wide):
    """Pawn pairs two files apart, with PP_3Wide's pair index over all 96 pawn ids.

    Complements PP_3Wide, which pairs pawns on the same or an adjacent file.
    """

    HASH = 0x86F2B1E2
    FEATURE_NAME = "PP_Wide2"
    INPUT_FEATURE_NAME = "PP_Wide2"
    # At most 8 pawns on each of two files two apart
    MAX_ACTIVE_FEATURES = 64
