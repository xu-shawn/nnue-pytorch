"""Reproducible single-node benchmark of the threats training configuration.

Run with torchrun --standalone --nproc-per-node=4 tests/bench_training.py DATA.
Timing includes the actual trainer, loss, optimizer, DDP and (by default) loader.
Use --cached to isolate compute with a ring of real, rank-sharded batches.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from data_loader.config import DataloaderSkipConfig
from model import NNUE
from model.config import LambdaConfig, LossParams, ModelConfig, NNUELightningConfig
from model.optimizers.config import OptimizerConfig
from train import make_data_loaders, _normalize_optimizer_and_schedulers
from trainer.callbacks import TerminateOnNaN, WeightClipper
from trainer.engine import SimpleTrainer, init_distributed


def threats_config():
    return NNUELightningConfig(
        features="Full_Threats+PP_3Wide+HalfKAv2_hm^",
        model_config=ModelConfig(L1=1024, L2=32),
        optimizer_config=OptimizerConfig(
            optimizer_name="adamw", lr=0.4e-3, factorized_weight_decay=4e-5,
            one_cycle_steps=5108000, one_cycle_warmup_pct=0.05,
            one_cycle_final_div=1e3,
        ),
        loss_params=LossParams(
            pow_exp=2.4340402395048404, qp_asymmetry=0.22866886710086187,
            in_scaling=295.6539508488627, out_scaling=379.98724077106635,
            in_offset=285.2706341467852, out_offset=289.1258344152218,
            lambda_config=LambdaConfig(
                lambda_cycle_jitter=True, jitter_lambda_sample=0.0035,
                jitter_lambda_batch=0.0070, jitter_decay_lambda_batch=0.999,
                start_lambda=1.0, end_lambda=1.0, lambda_cycle_warmup_pct=0.25,
                lambda_cycle_delta=-0.3, lambda_schedule_steps=5052000,
            ),
        ),
    )


class Timing:
    def __init__(self, warmup, steps, profile):
        self.warmup = warmup
        self.steps = steps
        self.profile = profile
        self.windows = []
        self.losses = []

    def on_train_batch_end(self, trainer, batch, batch_idx, outputs):
        step = batch_idx + 1
        if self.profile is not None:
            self.profile.step()
        if step >= self.warmup and (step - self.warmup) % self.steps == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            if step > self.warmup:
                self.windows.append(now - self.start)
                self.losses.append(float(outputs["loss"].detach()))
                if trainer.rank == 0:
                    print(f"Window {len(self.windows)}: {self.windows[-1]:.3f}s / {self.steps} steps", flush=True)
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.synchronize()
            self.start = time.perf_counter()


class CachedBatches:
    def __init__(self, batches, steps):
        self.batches = batches
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        for i in range(self.steps):
            yield self.batches[i % len(self.batches)]


class ProfiledLoader:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(len(self)):
            with torch.profiler.record_function("next_batch"):
                batch = next(iterator)
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="+", help="One or more training binpacks")
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--workers", "--num-workers", type=int, default=32, help="C++ threads per rank")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=512, help="Steps per timing window; use long windows for live loading")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cached", type=int, default=0, help="Number of real batches to cache on GPU")
    parser.add_argument("--batch-cache", help="Directory for reusable rank-local real batches (requires --cached)")
    parser.add_argument("--resume-from-checkpoint", help="Measure a trained model and optimizer state")
    parser.add_argument("--compile-backend", default="inductor", choices=["inductor", "cudagraphs", "eager"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", default="benchmark.json")
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.repeats, args.batch_size) <= 0:
        parser.error("warmup, steps, repeats and batch size must be positive")
    if args.workers <= 0 or args.threads <= 0 or args.cached < 0:
        parser.error("workers and threads must be positive; cached must be nonnegative")
    if args.batch_cache and not args.cached:
        parser.error("--batch-cache requires --cached")
    rank, world, local = init_distributed(device_type="cuda")
    if args.batch_size % world:
        parser.error("global batch size must be divisible by world size")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch._dynamo.config.cache_size_limit = 64
    device = torch.device("cuda", local)
    model = NNUE(config=threats_config(), max_epoch=4750, num_batches_per_epoch=1024)
    if args.compile_backend != "eager":
        model.model = torch.compile(model.model, backend=args.compile_backend)
    optimizer, schedulers = _normalize_optimizer_and_schedulers(model.configure_optimizers())
    total_steps = args.warmup + args.repeats * args.steps
    loader, _ = make_data_loaders(
        args.data, None, model.model.input_feature_name, args.workers,
        args.batch_size // world,
        DataloaderSkipConfig(
            random_fen_skipping=10, early_fen_skipping=18, soft_early_fen_skipping=32,
            pc_y0=-0.20, pc_y1=0.45, pc_y2=1.0, pc_y3=0.95, pc_y4=0.75,
            ply_x1=0.0, ply_y1=0.025, ply_x2=22.0, ply_y2=0.05,
            ply_x3=25.5, ply_y3=0.20, ply_x4=29.5, ply_y4=0.80,
        ), total_steps * args.batch_size, 0, True, 4,
        prefetch_device=device, rank=rank, world_size=world,
    )
    profiler = None
    if args.profile:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=args.warmup, warmup=1, active=3, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(f"{args.output}.rank{rank}.trace.json"),
        )
    timing = Timing(args.warmup, args.steps, profiler)
    trainer = SimpleTrainer(
        model=model, optimizer=optimizer, schedulers=schedulers, max_epochs=1,
        check_val_every_n_epoch=1, gradient_clip_val=2.0,
        log_every_n_steps=total_steps, default_root_dir=None,
        callbacks=[WeightClipper(), TerminateOnNaN(), timing], logger=None,
        device=device, rank=rank, world_size=world, local_rank=local,
    )
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
        trainer.max_epochs = trainer.current_epoch + 1
    if args.cached:
        cache = Path(args.batch_cache) / f"batch-{args.batch_size}-rank{rank}-of{world}.pt" if args.batch_cache else None
        if cache is not None and cache.exists():
            batches = torch.load(cache, map_location=device, weights_only=True)
            if len(batches) != args.cached:
                raise ValueError("Cached batch count does not match --cached")
        else:
            batches = [loader.dataset[i] for i in range(args.cached)]
            # Stop the producer so compute-only timing has no loader activity.
            loader.dataset._stop_prefetching.set()
            loader.dataset._prefetch_thread.join(timeout=10)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save([tuple(t.cpu() for t in b) for b in batches], cache)
        loader = CachedBatches(batches, total_steps)
    torch.cuda.reset_peak_memory_stats()
    if profiler is not None:
        loader = ProfiledLoader(loader)
        profiler.start()
    trainer.fit(loader)
    if not math.isfinite(trainer.callback_metrics["train_loss_epoch"]):
        raise RuntimeError("Non-finite loss: benchmark results are invalid")
    if profiler is not None:
        profiler.stop()
        if rank == 0:
            print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    elapsed = torch.tensor(timing.windows, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    elapsed = elapsed.cpu().tolist()
    result = {
        "args": vars(args), "world_size": world,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "window_seconds": elapsed,
        "positions_per_second": [args.batch_size * args.steps / t for t in elapsed],
        "median_step_ms": statistics.median(elapsed) * 1000 / args.steps,
        "losses_rank0": timing.losses,
        "peak_allocated_gb_rank0": torch.cuda.max_memory_allocated() / 1e9,
        "hostname": os.uname().nodename,
    }
    if rank == 0:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
