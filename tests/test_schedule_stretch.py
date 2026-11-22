import torch

from trainer.engine import SimpleTrainer

TOTAL_STEPS = 40


class _Metric:
    def compute(self):
        return torch.tensor(0.0)


class _Model(torch.nn.Module):
    """Records the step it is given, which is the one the lambda schedule uses."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.current_lambda = torch.ones(1)
        self.loss_metrics = {"train_loss_epoch": _Metric()}
        self.seen_steps = []

    def train_step(self, batch, current_epoch, step):
        self.seen_steps.append(step)
        loss = (self.weight * batch[0]).sum()
        return {"loss": loss, "train_loss": loss.detach()}


def _make(tmp_path, stretch, max_epochs):
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.0, total_steps=TOTAL_STEPS
    )
    trainer = SimpleTrainer(
        model=model,
        optimizer=optimizer,
        schedulers=[{"scheduler": scheduler, "interval": "step"}],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.0,
        log_every_n_steps=1000,
        default_root_dir=str(tmp_path),
        callbacks=[],
        logger=None,
        device="cpu",
        rank=0,
        world_size=1,
        local_rank=0,
        schedule_stretch=stretch,
    )
    return trainer, model, optimizer, scheduler


def _run(trainer, optimizer, steps_per_epoch):
    lrs = []

    class _Loader:
        def __iter__(self):
            for _ in range(steps_per_epoch):
                lrs.append(optimizer.param_groups[0]["lr"])
                yield (torch.ones(1),)

        def __len__(self):
            return steps_per_epoch

    trainer.fit(_Loader())
    return lrs


def test_default_steps_the_schedule_every_step(tmp_path):
    trainer, model, optimizer, scheduler = _make(tmp_path, 1.0, 1)
    _run(trainer, optimizer, 10)
    assert scheduler.last_epoch == 10
    assert trainer.schedule_step == trainer.global_step == 10
    assert model.seen_steps == list(range(10))


def test_stretch_after_resume_holds_each_value_twice(tmp_path):
    # Stage 1: 10 plain steps, saved as a checkpoint of the old format. The saved
    # epoch is 0, so each resumed trainer below runs one more epoch.
    trainer, _, optimizer, _ = _make(tmp_path, 1.0, 1)
    plain = _run(trainer, optimizer, 10)
    path = str(tmp_path / "stage1.ckpt")
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["schedule_step"]
    torch.save(checkpoint, path)

    # Reference: the same 10 further steps without a stretch.
    ref_trainer, _, ref_optimizer, _ = _make(tmp_path, 1.0, 1)
    ref_trainer.load_checkpoint(path)
    reference = _run(ref_trainer, ref_optimizer, 10)

    # Stage 2: twice as many steps with the schedules stretched by 2.
    trainer2, model2, optimizer2, scheduler2 = _make(tmp_path, 2.0, 1)
    trainer2.load_checkpoint(path)
    assert trainer2.schedule_step == 10.0
    stretched = _run(trainer2, optimizer2, 20)

    assert plain != reference
    assert stretched[0::2] == reference
    assert stretched[1::2] == reference
    assert scheduler2.last_epoch == 20
    assert trainer2.global_step == 30 and trainer2.schedule_step == 20.0
    assert model2.seen_steps == [10 + i // 2 for i in range(20)]

    # A checkpoint written mid-stretch keeps the schedule position.
    path2 = str(tmp_path / "stage2.ckpt")
    trainer2.save_checkpoint(path2)
    trainer3, _, _, scheduler3 = _make(tmp_path, 2.0, 1)
    trainer3.load_checkpoint(path2)
    assert trainer3.schedule_step == 20.0 and scheduler3.last_epoch == 20
