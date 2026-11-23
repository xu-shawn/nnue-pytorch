import torch

from model.config import LossParams
from model.nnue import calculate_sf_loss, group_loss


def _naive(qf, pf, outcome):
    total, rows = 0.0, list(range(len(outcome)))
    i = 0
    while i < len(rows):
        if outcome[i] > 2.5:
            j = i + 1
            while j < len(rows) and outcome[j] == 4.0:
                j += 1
            q, p = qf[i:j], pf[i:j]
            total += float((((q - q.mean()) - (p - p.mean())) ** 2).sum())
            i = j
        else:
            i += 1
    return total / len(rows)


def test_group_loss_matches_a_direct_computation():
    torch.manual_seed(0)
    # the tail of a group cut by the batch start, game rows, two whole groups,
    # a search-tree row, two groups back to back, a single-row group
    outcome = torch.tensor(
        [4, 4, 0.5, 1, 3, 4, 4, 2, 0, 3, 4, 3, 4, 4, 4, 0.5, 3], dtype=torch.float32
    ).view(-1, 1)
    qf, pf = torch.rand_like(outcome), torch.rand_like(outcome)
    expected = _naive(qf.view(-1), pf.view(-1), outcome.view(-1))
    assert abs(float(group_loss(qf, pf, outcome)) - expected) < 1e-7
    assert expected > 0


def test_group_loss_ignores_a_shift_of_a_whole_group():
    outcome = torch.tensor([3, 4, 4, 4], dtype=torch.float32).view(-1, 1)
    pf = torch.tensor([0.2, 0.4, 0.5, 0.7]).view(-1, 1)
    assert float(group_loss(pf + 0.1, pf, outcome)) < 1e-12
    assert float(group_loss(pf.flip(0), pf, outcome)) > 1e-3


def test_group_weight_zero_leaves_the_loss_unchanged():
    torch.manual_seed(1)
    outcome = torch.tensor([3, 4, 4, 0.5, 1, 2], dtype=torch.float32).view(-1, 1)
    scorenet, score = torch.randn(6, 1) * 300, torch.randn(6, 1) * 300
    lam = torch.tensor(1.0)
    base = calculate_sf_loss(scorenet, score, outcome, LossParams(), lam)
    weighted = calculate_sf_loss(
        scorenet, score, outcome, LossParams(group_weight=5.0), lam
    )
    plain_rows = calculate_sf_loss(
        scorenet,
        score,
        torch.tensor([2, 2, 2, 0.5, 1, 2], dtype=torch.float32).view(-1, 1),
        LossParams(group_weight=5.0),
        lam,
    )
    assert weighted > base
    assert abs(float(plain_rows) - float(base)) < 1e-9
