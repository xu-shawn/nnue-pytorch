# H100 investigation — 2026-09-26

Branch: `optimize/4xh100`, rebased onto
`eba1ed889d5a6a87d897362c829e7da204fbdfcc` (no PSQT). Existing changes from the
original branch were retained through the rebase.

**The new metric synchronization change improves measured compute/DDP throughput
by approximately 10.5%. A production sustained-training improvement has not been
established.** The rental's two-file loader bottleneck is not representative of
the supplied production environment. The rental was destroyed at the user's
request; no further rented resources are running for this task.

## Controlled results

All four-GPU measurements use global batch 131,072, or 32,768 per GPU, FP32,
AdamW, the threats loss and features, and the same trained no-PSQT checkpoint.
The compute comparisons replay eight real, rank-local batches already on the
GPUs. They include backward, optimizer updates, gradient clipping, and DDP;
they exclude live loading, transfers, validation, and checkpoint writing.

| Code / measurement | Step time | Iterations/s |
|---|---:|---:|
| Exact `eba1ed8`, four GPUs, cached real batches | ~16.4 ms | ~61 |
| Rebased original branch `748503e`, four GPUs, cached batches | 15.02 ms | 66.59 |
| Rebased branch plus metric change and NaN fix | 13.59 ms | 73.58 |
| Exact `eba1ed8`, one GPU, same **per-GPU** batch | 13.26 ms | 75.39 |
| Exact `eba1ed8`, live data, warm training epochs | ~27.8–28.4 ms | 36.11–36.85 |

The overall ~20% cached-batch improvement over `eba1ed8` includes changes already
present on the original branch. The additional ~10.5% is relative to the rebased
original branch, not to an unrelated or PSQT-enabled model. The one-GPU number
processes one quarter as many positions per iteration as the four-GPU number.

The optimized cached run used 64 warmup steps and three 256-step windows; its
window times were 3.4791, 3.4810, and 3.4756 seconds. The rebased control used
three 512-step windows: 7.6934, 7.6661, and 7.6890 seconds. These are unprofiled
timings. The earlier exact-base diagnostic included a three-step profiler
capture in its first window; only its subsequent windows support the approximate
61 it/s figure. Initialization and compilation are excluded.

## Profile findings

The exact-base, cached-data trace recorded approximately these GPU times per
step on rank zero:

| Work | Time |
|---|---:|
| Feature-transformer backward | 5.2 ms |
| Feature-transformer forward | 1.7 ms |
| NCCL all-reduces | 2.8 ms |
| AdamW | 1.3 ms |

Communication largely followed FT backward, with little overlap. Comparing one
and four GPUs at the same per-GPU batch gave about 3 ms additional step time.
These are diagnostic measurements, not a complete additive accounting of a step.

The three-step live-data trace happened to capture already-queued batches.
Longer measurements subsequently found 11–16 ms average host batch waits in
the worker-split experiment. This supersedes the initial suggestion that host
launch overhead alone explained the live-data slowdown.

## Changes retained

- Disable `MeanMetric`'s per-update NaN checks, which synchronize GPU work with
  the CPU. Epoch means still propagate non-finite losses. Require the tested
  `torchmetrics>=1.8.2` interface.
- Make every DDP rank enter the deferred non-finite-loss collective. Previously,
  a failure on just one rank could leave it waiting for other ranks.
- Fix CPU BMI2 detection: the old substring match also matched `avx512_vbmi2`,
  incorrectly disabling BMI2 on the rented EPYC CPUs. This is an x86 build fix;
  it is not relevant to the production Grace CPUs and produced no demonstrated
  sustained speedup here.
- Add the pinned threats launch configuration and a benchmark that records
  timings, source/library hashes, NCCL version, affinity, and batch waits.
- Extend fused-FT numerical coverage to empty/unequal feature rows, partial
  tiles, and L1 sizes up to 4096, including the rebased CUDA thread-limit fix.

## Experiments not retained

Skipping zero-valued gradient atomics did not improve cached throughput.
Compiling the loss did not materially improve sustained training. Forcing
`NCCL_PROTO=Simple` produced essentially the same cached throughput (~74 it/s).
None of these experiments is enabled in the delivered branch.

On this rental, 16 workers per GPU gave 18.79 it/s, 32 gave approximately
35–36 it/s, and 64 gave 37.94 it/s. Increasing the feature-worker fraction from
0.14 to 0.35 at 32 total workers gave 37.24 it/s. These loader experiments do
not justify changing production defaults. The original split and 32-worker
launch setting are retained. A PGO build was completed, but its training
measurement was canceled; no PGO speedup is claimed.

## Environment and reproduction

Rental: Vast instance 52810750, four H100 SXM 80 GB GPUs, NV6 peer connectivity,
dual EPYC 9554 CPUs, 128 physical CPU cores total, 512 GB RAM. Software:
Python 3.12, PyTorch 2.6.0+cu124, NCCL 2.21.5, CuPy 13.6, torchmetrics 1.9.0,
driver 550.163.01. Rental rate was approximately $6.96/hour.

Two complete binpacks from the configured corpus were downloaded **directly to
the rental**, and their SHA256 hashes were verified. Names, sizes, and hashes
are recorded in [datasets.json](benchmarks/h100-2026-09-26/datasets.json).
Before the no-PSQT benchmarks, exact `eba1ed8` completed all 1,024 training steps
of the configured 134,217,728-position epoch. Its checkpoint had finite weights,
the expected FT shapes, no PSQT parameters, and optimizer/scheduler state at
step 1,024. Additional actual training completed four epochs. Checkpoint loading
and resumed optimizer training were exercised throughout the later benchmarks.

The supplied [production log](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/5137461961076608/2926829081096545/-/jobs/16686853761/viewer)
confirms the same `eba1ed8`, batch, feature, loss, and 32-worker settings. It uses
an aarch64 image on Clariden, 72 CPU cores per rank, a PGO-built loader, and the
full binpack set. [CSCS documents the four Grace-Hopper modules per node](https://docs.cscs.ch/software/uenv/deploy/).
These are material differences from the rental. The available production log
contains many ~71–73 it/s epochs as well as slower epochs; it is not an
apples-to-apples baseline for the two-file rental runs.

An eight-second warm-run I/O sample showed ~360 MB/s logical reads per rank,
zero physical reads, and zero I/O pressure. The container reported no CPU quota
throttling. This rules out physical disk reads as the limiting factor during
that sample, not all possible differences between the machines.

## Validation and remaining limits

The rebased optimized implementation passed 59 tests with one MPS-only skip,
including CUDA forward/backward parity, loss metrics, NaN handling, lambda
scheduling, loader checks, and optimizer checks. The final metric/NaN-only tests
were also run locally after shutdown. A full validation cycle on the final
candidate and production/Elo validation were not completed before shutdown.

Selected raw results and reproduction/test logs are under
[benchmarks/h100-2026-09-26](benchmarks/h100-2026-09-26). The complete diagnostic
archive, including Chrome traces and rejected experiments, was saved locally
at `/tmp/nnue-h100/h100-results.tar.gz` before destruction. Earlier PSQT
experiments in that archive are obsolete and excluded from this report.

To reproduce the intended workload on a provisioned machine, build the loader
using `setup_script.sh`, then run:

```sh
bash scripts/train_threats_h100.sh /data/first.binpack /data/second.binpack \
  --default-root-dir /runs/threats
```

The script pins nettest configuration commit
`9bc8d4f4f89b9126079176dfdcd901d7bf294f7d`. Benchmark timing windows should be
unprofiled for performance comparisons; use `--profile` separately for diagnosis.
