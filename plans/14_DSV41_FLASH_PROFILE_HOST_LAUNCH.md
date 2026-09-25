# DeepSeek V4.1 Flash Host Launch Overhead Plan

> One of six profile-driven plans (09–14) from the MI355X C32 deep trace of
> 2026-09-24. The source profile, fixed baseline and performance rules are
> shared and kept once, in [Plan 09](./09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md)
> §0, §3 and §5.
> This is an internal execution document, not published InferenceX documentation.

Status: **not started**

> **Base changed (2026-09-24)** to `arbor-v2-plus-v6best-pr19` (Plan 09 §3).
> The call sites in §2.2 were read at `93205f244`. Recheck them on the new base
> before execution.

## 1. Objective

Find out whether host-side HIP runtime cost is on the critical path. If it is,
remove it. This is the only one of plans 09–14 that is not a kernel change.

## 2. Evidence

The handoff reports **3.22 ms of host time per step** excluding blocking sync,
against B200 C32's **1.26 ms**. That is 25% of a 12.90 ms step. Those
per-step figures divide by 861. The MI355X trace has 400 `ProfilerStep#`
annotations (Plan 09 §0), so its absolute values are 6.93 ms of host time in a
27.76 ms step. The 25% ratio is unchanged. The B200 figure has not been
rechecked against its own step count.

| API | MI355X deep | B200 C32 | per-call ratio | derived ms/step |
| --- | --- | --- | ---: | ---: |
| `hipGraphLaunch` | 1502 µs × 1,107 | `cudaGraphLaunch` 337 µs × 79 | 4.5× | **4.16** |
| `hipStreamIsCapturing` | 40.74 µs × 8,444 | `cudaStreamIsCapturing` 0.23 µs × 324 | 177× | 0.86 |
| `hipModuleLaunchKernel` | 7.19 µs × 40,562 | `cuLaunchKernelEx` 3.00 µs × 1,524 | 2.4× | 0.73 |
| `hipLaunchKernel` | 6.67 µs × 38,361 | `cudaLaunchKernel` 2.84 µs × 3,405 | 2.3× | 0.64 |
| `hipPointerGetAttribute` | 0.26 µs × 242,831 | `cudaPointerGetAttributes` 1.17 µs × 60 | 4,047× the calls | 0.16 |
| `hipStreamGetCaptureInfo_v2` | 40.84 µs × 507 | `cudaStreamGetCaptureInfo` 0.48 µs × 206 | 85× | 0.05 |

The ms/step column is derived here as calls × µs/call ÷ 400 profiler steps.
It is not in the source analysis.

What the table shows:

- **Graph launch dominates.** `hipGraphLaunch` is about 60% of host time per
  step. The trace-analysis note leads with `hipStreamIsCapturing` because its
  per-call ratio is the most anomalous, but graph launch is the larger cost.
- **`hipStreamIsCapturing` is still an anomaly.** At 40.74 µs it should be a
  register read. It reproduced at 42.27 µs in the shallow trace, so it is not
  noise.
- **MI355X may not be launching more.** The handoff reports similar kernel
  counts per step (875 against 963 on B200 C32) and fewer eager launches, but
  both counts share the 861-step denominator. On 400 steps MI355X runs about
  1,880 kernels per step. Recheck B200's step count before comparing launch
  counts.

### 2.1 Host time only matters if the GPU waits for it

The shallow C32 TraceLens timeline shows **7.38% GPU idle** on MI355X against
7.50% on B200. Similar idle despite 2.5× the host time suggests most host cost
is hidden behind GPU work at C32. Two caveats:

- that window was shallow and short, and idle was never computed on the deep
  trace;
- C8 does less GPU work per step, so host exposure there is likely larger, and
  no MI355X C8 trace exists.

A long `hipGraphLaunch` can also be benign back-pressure: the call blocks
because the hardware queue is full and the GPU is busy. That is waiting, not
overhead. Step 4.1 decides between the two before anything is changed.

### 2.2 Hot-path capture queries in this tree

On ROCm, `torch.cuda.is_current_stream_capturing()` becomes a
`hipStreamIsCapturing` call. At `93205f244` it is called on hot paths in:

- `vllm/_aiter_ops.py:1052` and `:1104`: the fused AR + RMSNorm + quant op,
  on every call, to pick `registered=`;
- `vllm/distributed/device_communicators/custom_all_reduce.py:447`;
- `vllm/model_executor/layers/sparse_attn_indexer.py:439`;
- `vllm/models/deepseek_v4_1/attention.py:1197`.

Inside graph replay these queries do not run. They cost time on eager steps
(prefill and mixed batches) and during capture.

## 3. Fixed baseline

As Plan 09 §3, including `VLLM_USE_BREAKABLE_CUDAGRAPH=1` from the env file.
Only host-side runtime usage may change. No kernel changes.

## 4. Work items

### 4.1 Exposed host time

For each GPU idle gap in the deep trace, record which host API call is in
flight. From that, report exposed host ms per step at C32, and at C8 from the
shared C8 capture (Plan 09 §4.1). Classify each `hipGraphLaunch` as blocking
back-pressure (GPU busy for the whole call) or overhead (GPU idle during it).

If exposed host time is small at both points, stop (§6). Nothing in this plan
would then move throughput.

### 4.2 Graph launch

- Measure `hipGraphLaunch` cost against graph node count.
- Count graph launches per step. `VLLM_USE_BREAKABLE_CUDAGRAPH=1` splits graphs,
  so it may multiply launches.
- Only if 4.1 shows graph launch exposed: evaluate fewer, larger graphs, and
  whether DSV4.1 runs correctly without breakable graphs. That is a
  configuration change, measured like any other candidate.

### 4.3 `hipStreamIsCapturing` microbenchmark

In the benchmark image, time the call in isolation, then with a second thread
issuing `hipGraphLaunch` and `hipModuleLaunchKernel`, then under four processes
(TP4). The comparison separates intrinsic cost from runtime-lock contention.
File a HIP issue with the reproducer either way, and record the ROCm and HIP
versions of the image.

### 4.4 Cached capture state, behind a knob

vLLM owns graph capture, so a process-level flag set by its capture context can
replace the hot-path queries in §2.2. Add this behind a knob, default off.

The risk is specific. A wrong `False` during capture passes `registered=False`
for an IPC buffer inside a graph, which can hang or corrupt the allreduce
(Plan 11). Before any performance run, unit tests must prove the flag is true
exactly within capture, including breakable-graph segments and warmup captures.

### 4.5 `hipPointerGetAttribute` (last)

The 242,831 calls cost about 63 ms per window. Find the caller and remove the
repeated queries only if that is trivial.

## 5. Gates

- **Correctness.** The unit tests in 4.4 pass, and a graph-replay soak runs with
  no hang, before any timed run.
- **Trace.** Host time per step and GPU idle both fall. A host reduction that
  leaves GPU idle unchanged is not a result.
- **Performance.** Plan 09 §5. C8 is the expected beneficiary. Report it first,
  and do not book a C32-only effect.

## 6. Stop conditions

- 4.1 finds little exposed host time at both C8 and C32.
- `hipGraphLaunch` turns out to be back-pressure. The GPU work is then the
  bottleneck, and plans 09–13 own it.
- Cached capture state cannot be proven exact.

## 7. Artifacts

```text
crusoe:/home/jiaweche/dsv41-profile-20260924/14_host_launch/
```

Persist the gap attribution, the microbenchmark with its reproducer, the HIP
issue link, graph-launch counts, the knob diff, trace captures, per-replay bench
outputs and the decision.
