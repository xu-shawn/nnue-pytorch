from .callbacks import WeightClippingCallback
from .config import L1, L2, L3, SetNetworkSize, LossParams
from .lightning_module import NNUE
from .model import NNUEModel
from .utils import coalesce_ft_weights

def add_argparse_args(parser):
    parser.add_argument("--l1", type=int, default=L1, action=SetNetworkSize)


__all__ = [
    "WeightClippingCallback",
    "L1",
    "L2",
    "L3",
    "LossParams",
    "NNUE",
    "NNUEModel",
    "coalesce_ft_weights",
]
