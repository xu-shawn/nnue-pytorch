"""Mixed precision must preserve small updates and checkpoint/skip semantics."""
import pytest
import torch
from torchmetrics import MeanMetric

from trainer.engine import SimpleTrainer


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(8, 1)
        self.loss_metrics = torch.nn.ModuleDict({"train_loss_epoch": MeanMetric()})
        self.forward_dtypes = []

    def train_step(self, batch, epoch, step):
        x, y = batch
        prediction = self.model(x)
        self.forward_dtypes.append(prediction.dtype)
        loss = (prediction.float() - y).square().mean()
        self.loss_metrics["train_loss_epoch"].update(loss.detach())
        return {"loss": loss, "train_loss": loss.detach()}


def make_trainer(device):
    torch.manual_seed(718)
    model = ToyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device=="cuda")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=10)
    trainer = SimpleTrainer(model, optimizer, [scheduler], 1, 1, 0.1, 100, None,
                            [], None, device, 0, 1, 0, mixed_precision=True)
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=device == "cuda")
    trainer._last_scale = trainer.scaler.get_scale()
    x = torch.randn(16, 8, device=device)
    y = torch.randn(16, 1, device=device)
    return trainer, [(x, y)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_precision_state_and_resume(device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    trainer, batches = make_trainer(device)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.fit(batches * 3)
    assert trainer.model.forward_dtypes == [torch.float16 if device=="cuda" else torch.float32]*3
    assert trainer.global_step == 3
    assert trainer.optimizer_steps_skipped == 0
    norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in trainer.model.parameters()]))
    torch.testing.assert_close(norm, norm.new_tensor(0.1), rtol=3e-5, atol=2e-6)
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters())
    assert any(not torch.equal(p, old) for p, old in zip(trainer.model.parameters(), before))
    for state in trainer.optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        assert torch.isfinite(state["exp_avg_sq"]).all()
    path = str(tmp_path / "resume.ckpt")
    trainer.save_checkpoint(path)
    restored, _ = make_trainer(device)
    restored.load_checkpoint(path)
    assert restored.scaler.state_dict() == trainer.scaler.state_dict()
    assert restored.global_step == trainer.global_step
    assert restored._schedulers[0].state_dict() == trainer._schedulers[0].state_dict()
    # Older FP32 checkpoints have no scaler state.
    state = torch.load(path, weights_only=False)
    state.pop("grad_scaler")
    torch.save(state, path)
    restored.load_checkpoint(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overflow_skips_update_and_scheduler():
    trainer, batches = make_trainer("cuda")
    before = [p.detach().clone() for p in trainer.model.parameters()]
    initial_epoch = trainer._schedulers[0].last_epoch
    handle = trainer.model.model.weight.register_hook(lambda g: g * float("inf"))
    trainer.fit(batches)
    handle.remove()
    assert trainer.optimizer_steps_skipped == 1
    assert trainer.global_step == 0
    assert trainer._schedulers[0].last_epoch == initial_epoch
    assert all(torch.equal(p, old) for p, old in zip(trainer.model.parameters(), before))
    assert trainer.scaler.get_scale() == 512
    trainer.fit(batches)
    assert trainer.global_step == 1
    assert trainer._schedulers[0].last_epoch == initial_epoch + 1
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
