import torch
from torch import nn

from ..quantize import QuantizationManager

# Material input: one-hot features of the material configuration, seen from the side to
# move, added to the outputs of the first layer-stack layer (shared by all stacks). With N
# = knights, B = bishops, R = rooks, Q = queens, P = pawns:
#   - per side (side to move, then opponent): min(count, cap) for P, N, B, R, Q with caps
#     8, 2, 2, 2, 1 (40 features)
#   - side to move minus opponent: N + B clamped to [-3, 3], R to [-2, 2], Q to [-1, 1],
#     P to [-4, 4] (24 features)
#   - bishop pair (bishops on both square colours), per side (2 features)
#   - joint non-pawn signature: per side s = min(Q, 1) * 27 + min(R, 2) * 9 + min(B, 2) * 3
#     + min(N, 2), feature s(side to move) * 54 + s(opponent) (2916 features)
# At most 17 features are active. The engine computes the same indices from the board.
CAPS = (8, 2, 2, 2, 1)  # P, N, B, R, Q
NUM_FEATURES = 2 * sum(c + 1 for c in CAPS) + 7 + 5 + 3 + 9 + 2 + 54 * 54  # 2982
MAX_ACTIVE = 17
HASH = 0x4D415431  # "MAT1", written before the parameters in the .nnue


class MaterialInput(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        psq_offset: int,
        psq_span: int,
        quantization: QuantizationManager,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.psq_offset = psq_offset
        self.psq_span = psq_span
        self.quantization = quantization
        self.linear = nn.Linear(NUM_FEATURES, num_outputs)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()

    def feature_indices(self, piece_square_indices: torch.Tensor) -> torch.Tensor:
        """Active material features [batch, MAX_ACTIVE] (-1 for unused slots) from the side
        to move's K32Q2 indices (index = square + piece * 64 + bucket * 768, where
        piece = piece type * 2 + (piece belongs to the opponent))."""
        idx = piece_square_indices.long()
        rel = idx - self.psq_offset
        valid = (idx >= 0) & (rel >= 0) & (rel < self.psq_span)
        plane = torch.where(valid, rel % 768, torch.zeros_like(rel))
        piece, square = plane // 64, plane % 64

        n = idx.shape[0]
        counts = torch.zeros(n, 13, dtype=torch.long, device=idx.device)
        counts.scatter_add_(
            1,
            torch.where(valid, piece, torch.full_like(piece, 12)),
            torch.ones_like(piece),
        )
        own, opp = counts[:, 0:10:2], counts[:, 1:10:2]  # P, N, B, R, Q

        # Bishops by square colour; any fixed board orientation flips the colour of every
        # square alike, so "bishops on both colours" does not depend on it
        colour = (square % 8 + square // 8) % 2
        is_bishop = valid & ((piece == 4) | (piece == 5))
        slot = torch.where(
            is_bishop, (piece - 4) * 2 + colour, torch.full_like(piece, 4)
        )
        bishops = torch.zeros(n, 5, dtype=torch.long, device=idx.device)
        bishops.scatter_add_(1, slot, torch.ones_like(slot))

        cols = []
        offset = 0
        for side in (own, opp):
            for t, cap in enumerate(CAPS):
                cols.append(offset + side[:, t].clamp(max=cap))
                offset += cap + 1
        diffs = (
            (own[:, 1] + own[:, 2] - opp[:, 1] - opp[:, 2], 3),
            (own[:, 3] - opp[:, 3], 2),
            (own[:, 4] - opp[:, 4], 1),
            (own[:, 0] - opp[:, 0], 4),
        )
        for d, lim in diffs:
            cols.append(offset + d.clamp(-lim, lim) + lim)
            offset += 2 * lim + 1
        for s in range(2):
            pair = (bishops[:, 2 * s] > 0) & (bishops[:, 2 * s + 1] > 0)
            cols.append(
                torch.where(
                    pair, offset + s, torch.full_like(pair, -1, dtype=torch.long)
                )
            )
        offset += 2

        def signature(side):
            return (
                side[:, 4].clamp(max=1) * 27
                + side[:, 3].clamp(max=2) * 9
                + side[:, 2].clamp(max=2) * 3
                + side[:, 1].clamp(max=2)
            )

        cols.append(offset + signature(own) * 54 + signature(opp))
        offset += 54 * 54
        assert offset == NUM_FEATURES and len(cols) == MAX_ACTIVE
        return torch.stack(cols, dim=1)

    def forward(
        self, feature_indices: torch.Tensor, fake_quantize_weights: bool
    ) -> torch.Tensor:
        weight, bias = self.linear.weight, self.linear.bias
        if fake_quantize_weights:
            weight = self.quantization.fake_quantize_weights(weight, "material")
            bias = self.quantization.fake_quantize_weights(bias, "material")
        rows = weight.t()[feature_indices.clamp(min=0)]  # [batch, MAX_ACTIVE, outputs]
        rows = rows * (feature_indices >= 0).unsqueeze(-1).to(rows.dtype)
        return rows.sum(dim=1) + bias
