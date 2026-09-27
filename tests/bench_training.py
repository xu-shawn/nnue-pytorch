"""Reproducible single-node benchmark of the threats training configuration.

Run with torchrun --standalone --nproc-per-node=4 tests/bench_training.py DATA.
Timing includes the actual trainer, loss, optimizer, DDP and (by default) loader.
Use --cached to isolate compute with a ring of real, rank-sharded batches.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from data_loader.config import DataloaderSkipConfig
from data_loader._native import c_lib
from model import NNUE
from model.config import LambdaConfig, LossParams, ModelConfig, NNUELightningConfig
from model.optimizers.config import OptimizerConfig
from train import make_data_loaders, _normalize_optimizer_and_schedulers
from trainer.callbacks import TerminateOnNaN, WeightClipper
from trainer.engine import SimpleTrainer, init_distributed


def threats_config(grouped_l1=False, sparse_l1=False):
    return NNUELightningConfig(
        features="Full_Threats+PP_3Wide+HalfKAv2_hm^",
        model_config=ModelConfig(L1=1024, L2=32, grouped_l1=grouped_l1, sparse_l1=sparse_l1),
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


def source_metadata():
    root = Path(__file__).resolve().parents[1]
    paths = [
        "config.py", "train.py", "scripts/train_threats_h100.sh",
        "model/nnue.py", "trainer/engine.py", "trainer/callbacks.py",
        "model/modules/config.py", "model/modules/layer_stacks.py",
        "model/modules/stacked_linear.py", "model/modules/grouped_linear.py",
        "model/modules/feature_transformer/fused_ft_kernel.py",
        "model/modules/feature_transformer/fused_ft_functions.py",
        "model/modules/feature_transformer/aggregated_ft_kernel.py",
        "tests/bench_training.py",
    ]
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    library = Path(c_lib.dll._name)
    hashes["loader_library"] = hashlib.sha256(library.read_bytes()).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
    )
    return {"revision": revision.stdout.strip() or None, "sha256": hashes}


class Timing:
    def __init__(self, warmup, steps, profile, nsys=False):
        self.warmup = warmup
        self.steps = steps
        self.profile = profile
        self.nsys = nsys
        self.windows = []
        self.losses = []

    def on_train_batch_start(self, trainer, batch, batch_idx):
        if self.nsys and batch_idx >= self.warmup:
            torch.cuda.nvtx.range_push(f"train_step_{batch_idx}")

    def on_train_batch_end(self, trainer, batch, batch_idx, outputs):
        step = batch_idx + 1
        if self.nsys and step > self.warmup:
            torch.cuda.nvtx.range_pop()
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
            if self.nsys and step == self.warmup:
                torch.cuda.cudart().cudaProfilerStart()
            elif self.nsys and step == self.warmup + self.steps:
                torch.cuda.cudart().cudaProfilerStop()
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
    def __init__(self, loader, warmup, profile):
        self.loader = loader
        self.warmup = warmup
        self.profile = profile
        self.wait_seconds = []

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        for i in range(len(self)):
            start = time.perf_counter()
            if self.profile:
                with torch.profiler.record_function("next_batch"):
                    batch = next(iterator)
            else:
                batch = next(iterator)
            if i >= self.warmup:
                self.wait_seconds.append(time.perf_counter() - start)
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="+", help="One or more training binpacks")
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--workers", "--num-workers", type=int, default=32, help="C++ threads per rank")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=50)
    parser.add_argument("--grouped-l1", action="store_true")
    parser.add_argument("--sparse-l1", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=512, help="Steps per timing window; use long windows for live loading")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cached", type=int, default=0, help="Number of real batches to cache on GPU")
    parser.add_argument("--random-fen-skipping", type=int, default=10,
                        help="Use 0 to diagnose loader starvation; changes training sampling")
    parser.add_argument("--unfiltered", action="store_true",
                        help="Disable the entire native skip predicate for loader diagnostics")
    parser.add_argument("--save-checkpoint", help="Save trained state after timing for controlled subsequent runs")
    parser.add_argument("--batch-cache", help="Directory for reusable rank-local real batches (requires --cached)")
    parser.add_argument("--resume-from-checkpoint", help="Measure a trained model and optimizer state")
    parser.add_argument("--compile-backend", default="inductor", choices=["inductor", "cudagraphs", "eager"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--nsys", action="store_true",
                        help="Capture the first warmed window with Nsight Systems cudaProfilerApi ranges")
    parser.add_argument("--output", default="benchmark.json")
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.repeats, args.batch_size) <= 0:
        parser.error("warmup, steps, repeats and batch size must be positive")
    if args.workers <= 0 or args.threads <= 0 or args.cached < 0 or args.random_fen_skipping < 0:
        parser.error("workers and threads must be positive; cached and random skipping must be nonnegative")
    if args.batch_cache and not args.cached:
        parser.error("--batch-cache requires --cached")
    if args.ddp_bucket_cap_mb <= 0:
        parser.error("--ddp-bucket-cap-mb must be positive")
    if args.nsys and (args.profile or args.repeats != 1):
        parser.error("--nsys requires --repeats 1 and cannot be combined with --profile")
    rank, world, local = init_distributed(device_type="cuda")
    if args.batch_size % world:
        parser.error("global batch size must be divisible by world size")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch._dynamo.config.cache_size_limit = 64
    device = torch.device("cuda", local)
    model = NNUE(config=threats_config(args.grouped_l1, args.sparse_l1), max_epoch=4500, num_batches_per_epoch=1024)
    if args.compile_backend != "eager":
        model.model = torch.compile(model.model, backend=args.compile_backend)
    optimizer, schedulers = _normalize_optimizer_and_schedulers(model.configure_optimizers())
    total_steps = args.warmup + args.repeats * args.steps
    skip_config = DataloaderSkipConfig(
        random_fen_skipping=args.random_fen_skipping,
        early_fen_skipping=18, soft_early_fen_skipping=32,
        pc_y0=-0.20, pc_y1=0.45, pc_y2=1.0, pc_y3=0.95, pc_y4=0.75,
        ply_x1=0.0, ply_y1=0.025, ply_x2=22.0, ply_y2=0.05,
        ply_x3=25.5, ply_y3=0.20, ply_x4=29.5, ply_y4=0.80,
    )
    if args.unfiltered:
        # This combination makes make_skip_predicate return nullptr, bypassing
        # all filtering, including the adaptive piece-count filter.
        skip_config = DataloaderSkipConfig(
            filtered=False, wld_filtered=False, random_fen_skipping=0,
            early_fen_skipping=-1, soft_early_fen_skipping=0,
        )
    loader, _ = make_data_loaders(
        args.data, None, model.model.input_feature_name, args.workers,
        args.batch_size // world, skip_config, total_steps * args.batch_size, 0, True, 4,
        prefetch_device=device, rank=rank, world_size=world,
    )
    profiler = None
    if args.profile:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=args.warmup, warmup=1, active=3, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(f"{args.output}.rank{rank}.trace.json"),
        )
    timing = Timing(args.warmup, args.steps, profiler, args.nsys)
    trainer = SimpleTrainer(
        model=model, optimizer=optimizer, schedulers=schedulers, max_epochs=1,
        check_val_every_n_epoch=1, gradient_clip_val=2.0,
        log_every_n_steps=total_steps, default_root_dir=None,
        callbacks=[WeightClipper(), TerminateOnNaN(), timing], logger=None,
        device=device, rank=rank, world_size=world, local_rank=local,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
    )
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
        trainer.max_epochs = trainer.current_epoch + 1
    if args.cached:
        cache = Path(args.batch_cache) / f"batch-{args.batch_size}-rank{rank}-of{world}.pt" if args.batch_cache else None
        cache_metadata = {
            "features": model.model.input_feature_name,
            "skip_config": asdict(skip_config),
            "data": [{"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size} for p in args.data],
            "batch_size": args.batch_size, "world_size": world,
            "rank": rank, "count": args.cached,
        }
        if cache is not None and cache.exists():
            metadata_path = cache.with_suffix(".json")
            if not metadata_path.exists() or json.loads(metadata_path.read_text()) != cache_metadata:
                raise ValueError("Batch cache metadata differs from this workload; use a separate cache directory")
            batches = torch.load(cache, map_location=device, weights_only=True)
            if len(batches) != args.cached:
                raise ValueError("Cached batch count does not match --cached")
        else:
            batches = [loader.dataset[i] for i in range(args.cached)]
            # Stop the producer so compute-only timing has no loader activity.
            loader.dataset._stop_prefetching.set()
            loader.dataset._prefetch_thread.join(timeout=10)
            if loader.dataset._prefetch_thread.is_alive():
                raise RuntimeError("Loader producer did not stop; cached timing would be contaminated")
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save([tuple(t.cpu() for t in b) for b in batches], cache)
                cache.with_suffix(".json").write_text(json.dumps(cache_metadata, indent=2) + "\n")
        loader = CachedBatches(batches, total_steps)
    torch.cuda.reset_peak_memory_stats()
    loader = ProfiledLoader(loader, args.warmup, args.profile)
    if profiler is not None:
        profiler.start()
    trainer.fit(loader)
    if args.save_checkpoint:
        trainer.save_checkpoint(args.save_checkpoint)
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
    waits = sorted(loader.wait_seconds)
    wait_stats = {
        "rank": rank,
        "mean_ms": statistics.mean(waits) * 1000,
        "p95_ms": waits[int(0.95 * (len(waits) - 1))] * 1000,
        "max_ms": max(waits) * 1000,
    }
    rank_wait_stats = [None] * world
    if world > 1:
        dist.all_gather_object(rank_wait_stats, wait_stats)
    else:
        rank_wait_stats[0] = wait_stats
    result = {
        "args": vars(args), "world_size": world,
        "skip_config": asdict(skip_config),
        "source": source_metadata(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "cpu_affinity_rank0": sorted(os.sched_getaffinity(0)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "nccl_environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        "allocator_environment": {k: v for k, v in os.environ.items() if k.startswith("MALLOC_") or k == "GLIBC_TUNABLES"},
        "experiment_environment": {k: v for k, v in os.environ.items() if k.startswith("NNUE_")},
        "python": sys.version,
        "python_switch_interval": sys.getswitchinterval(),
        "gpu": torch.cuda.get_device_name(),
        "window_seconds": elapsed,
        "host_next_batch_wait": rank_wait_stats,
        "iterations_per_second": [args.steps / t for t in elapsed],
        "profile_overhead_in_window": 0 if args.profile else None,
        "nsys_profiled": args.nsys,
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
