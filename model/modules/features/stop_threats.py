from .pp_3wide import PP3Wide


class StopThreats(PP3Wide):
    """Threat features for the empty square in front of every pawn (its stop square).

    Every knight, bishop, rook, queen or king attacking it is a feature of its own, indexed
    by pawn color, pawn square, attacker kind (own knight, bishop, rook, queen, king, then
    the enemy's) and attacker square, counting only the squares from which that kind can
    reach the stop square. Pawn attackers are covered by PP_3Wide.
    """

    HASH = 0x7D29A4F1
    FEATURE_NAME = "Stop_Threats"
    INPUT_FEATURE_NAME = "Stop_Threats"
    # At most 18 non-pawn attackers per square and 16 pawns
    MAX_ACTIVE_FEATURES = 16 * 18

    NUM_INPUTS = 11256
    NUM_REAL_FEATURES = NUM_INPUTS
