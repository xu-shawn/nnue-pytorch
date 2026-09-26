import pytest
import torch

from model import NNUE, NNUELightningConfig
from model.config import ModelConfig


@pytest.fixture
def metrics():
    return NNUE(
        config=NNUELightningConfig(model_config=ModelConfig(L1=32, L2=4))
    ).loss_metrics


def test_loss_metrics_average_and_reset(metrics):
    for metric in metrics.values():
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(3.0))
        torch.testing.assert_close(metric.compute(), torch.tensor(2.0))
        metric.reset()
        metric.update(torch.tensor(5.0))
        torch.testing.assert_close(metric.compute(), torch.tensor(5.0))


@pytest.mark.parametrize("loss", [float("nan"), float("inf")])
def test_loss_metrics_preserve_nonfinite_losses(metrics, loss):
    # Invalid batches must remain visible in epoch metrics, rather than
    # disappearing from the mean when per-update host checks are disabled.
    for metric in metrics.values():
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(loss))
        assert not torch.isfinite(metric.compute())
