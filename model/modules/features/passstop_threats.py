from .stop_threats import StopThreats


class PassStopThreats(StopThreats):
    """Stop_Threats restricted to passed pawns, with Stop_Threats' index."""

    HASH = 0x7D29A4F7
    FEATURE_NAME = "PassStop_Threats"
    INPUT_FEATURE_NAME = "PassStop_Threats"
