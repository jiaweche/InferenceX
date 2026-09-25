# DeepSeek V4.1 Flash Host Launch Overhead Plan

> One of six profile-driven plans (09–14) from the MI355X C32 deep trace of
> 2026-09-24. The source profile, fixed baseline and performance rules are
> shared and kept once, in [Plan 09](./09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md)
> §0, §3 and §5.
> This is an internal execution document, not published InferenceX documentation.

Status: **stopped** by the operator on 2026-09-25 and replaced by Plan 16. It
had been submitted to the Plan 11 executor mesh (setup in §8). One of its ten
tasks was done (graph-launch counts from Plan 10's rank-0 captures); see
`crusoe:/home/jiaweche/dsv41-profile-20260924/14_host_launch/CANCELLED.md`.

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

## 8. Execution setup

Written by the operator before submission. These are facts about the
environment, not choices about the plan.

### 8.1 Mesh, node and trees

- **Mesh.** This plan runs on the mesh that executed Plans 11 and 13: session
  `/home/jiaweche/Arbor-plan-executor-p11/sessions/plan-executor`, keeper
  `arbor-p11-keeper` on `slog-007`. Both earlier plans settled with stop, and
  both reviews are done.
- **Node.** Job `174831` on `crsuse2-m2m-016` (`amd-burst`, 24 h, preemptible).
  The pool adopts jobs named `arbor-p11-*`. All four job slots of the account
  are in use.
- **A fresh container.** At 20:46 UTC on 2026-09-25, `dsv41flash_arbor` was
  recreated from the image by digest (`sha256:960228cf…`), and the fixed base
  `be794db46` was overlaid again. The old container had held Plan 13's work, and
  an orphaned vLLM server (four `VLLM::Worker_TP*` processes, started 19:29 UTC,
  with its API server dead) was still holding GPUs 0–3 after Plan 13 settled.
  **Stop every server you start before a task closes.**
  - AITER is stock 0.1.21.post2, and its JIT caches start cold.
  - Weights are node-local at
    `/mnt/m2m_nobackup/jiaweche/inferencex-dsv41flash/models/DeepSeek-V4.1-Flash`.
  - The aiperf client is at `/runtime/aiperf-src`.
  - The setup log is `$R/logs/setup.log`.
- **Other tenants' containers** (`miles_primus_yuankai`,
  `oshkarav-inference-testing-amd-1`) sit on this node. Neither held a GPU at
  setup. Record GPU occupancy before every measured stage, as Plan 09's
  `measure_one.sh` does.
- **Spur runs job steps in their own PID namespace.** Inside
  `srun --overlap`, PID 1 is the job script, so host PIDs reported by
  `amd-smi process` are not in `/proc`. Map them to containers with
  `docker top <container> -eo pid,lstart,args`.
- **Not ours.** Never touch these jobs, their containers, or their sessions.
  Read-only access to their files is fine.

  | Job | Node | Belongs to | Mesh runs on | Session |
  | --- | --- | --- | --- | --- |
  | `174800` | `m2m-032` | Plan 10 | `slog-005` | `/home/jiaweche/Arbor-plan-executor` |
  | `174801` | `m2m-191` | Plan 12 | `slog-006` | `/home/jiaweche/Arbor-plan-executor-p12` |
  | `174972` | `m2m-193` | Plan 15 | `slog-006` | `/home/jiaweche/Arbor-plan-executor-p15` |

- **Paths.**
  - Artifacts (§7): `R=/home/jiaweche/dsv41-profile-20260924/14_host_launch`.
  - Base checkout: `/home/jiaweche/dsv41-merge/vllm-v6best-pr19`, a worktree of
    `/home/jiaweche/dsv41-merge/vllm`. Do not commit on it.
  - This plan's branch: make a new worktree from `be794db46`, for example
    branch `arbor-v2-plus-v6best-pr19-p14` at
    `/home/jiaweche/dsv41-merge/vllm-p14`.
- **Deploying Python changes.** Run
  `SRC=<worktree> WANT_SHA=<full sha> bash $R/bin/deploy_tree.sh` on the node.
  It overlays `vllm/`, hash-checks every `.py` in the commit against the
  container, confirms the compiled extensions are untouched, and imports the
  DSA attention modules. Anything short of `DEPLOY_OK sha=<sha>` means the tree
  is not what runs. §3 allows only host-side runtime changes, so no compiled
  deploy should be needed.
- **Server configuration.** The control server's env file (§3's
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1` comes from it) is `server_env_blind.list`
  in the v9 toolbox named in §8.2, md5 `64a08abe4e522e6ba451e4b6bebdebcb`
  (Plan 09 §3).
- **MoE configs.** The base's PR #19 MoE-config install writes its CSV into the
  AITER package inside the container, so every server after the first runs the
  FlyDSL MoE configs. That is identical in both arms of every A/B here and must
  stay so. Plan 09 owns it.
- **GPUs.** A server runs TP4 on GPUs 0–3. GPUs 4–7 are idle, which is where
  §4.3's isolated microbenchmark can run without disturbing a server.
- **Other plans.** Plans 09, 11 and 15 are in
  `/home/jiaweche/dsv41-profile-20260924/plans/`.

### 8.2 Existing work to reuse

- **Plan 15** runs on its own mesh (`slog-006`). Its §4.0 is capturing the base
  at C8 and C32 on **all four ranks**, to measure allreduce arrival spread. It
  hands host-gap findings to this plan (its §6). Its artifacts are
  `/home/jiaweche/dsv41-profile-20260924/15_collectives_followup/`: read, never
  write. If those four-rank captures exist when §4.1 starts, they record host
  runtime calls on every rank. Captures compared against a candidate are
  still taken on this node.
- **Rank-0 captures of this base.**
  - Plan 10's knob-off captures at C32 and C8, 400 `ProfilerStep#` each:
    `/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/cap_base_c{32,8}/traces/`.
  - Plan 09's captures:
    `/home/jiaweche/dsv41-profile-20260924/09_moe_experts/run1/trace_a{0,1}_c{8,32}/traces/`.
- **Plan 11's findings on the capture-time hazard** that §4.4 names: its
  settlement, review and TODO are in `.../11_tp_allreduce/T09_SETTLEMENT.md`
  and this mesh's session (`mesh/reviews/p01-reviewer.md`, `mesh/TODO.md`).
- **Scripts.** Copy and change what differs:
  - Plan 09 (`.../09_moe_experts/run1/bin/`): `capture_one.sh`,
    `measure_one.sh` (one A/B point: fresh control server, marker check,
    discarded warmup, one measured replay), `p09_driver.sh` and
    `summarize_ab.py`.
  - Plan 10 (`.../10_dsa_indexer_decode/run1/bin/`): `dsa_attrib.py` (Kineto
    attribution, including graph-replayed kernels via the `correlation` of their
    `hipGraphLaunch`) and `t01_driver.sh` (captures under a user unit).
- **Harness.** Read, never edit; it belongs to another mesh's session:
  `/home/jiaweche/Arbor-dsv41-mesh/sessions/megamoe-v9/mesh/toolbox/crusoe/{boot_control_server.sh,replay_arm.sh}`.
- **The Ruby deep trace** behind §2 (and §4.1's "deep trace") is on `ruby-1`,
  which Crusoe cannot reach, so §2's numbers are the reference, not a file.

### 8.3 Things that have already cost time here

- **What Plan 11's review found.** The reviewer upheld Plan 11's stop but
  rejected several of its conclusions. Avoid repeating these:
  - **A gate that was not run is "not run".** It is not "passed", even after a
    stop makes it moot.
  - **Keep the raw samples.** Keep each microbenchmark replay's samples, and
    record the exact script that produced them.
  - **Step counts come from `ProfilerStep#`.** Every capture here records 400.
    Never infer them from median latency.
  - **Timing overlap is not independence.** Two kernels not overlapping today
    says nothing about whether one depends on the other.
- **When a settled plan's work is done, it is done.** After Plan 13 settled,
  this mesh's agents spent more than an hour messaging one another about its
  records. Correct a settled record once, where it lives, and move on.
- **Memory.** Parse traces on the node through
  `srun --overlap --jobid=174831`, never on the login node.
  - An agent that parsed a Kineto trace on a login node grew to 1.8 GB, and
    every process of this user on that host slowed to a crawl.
  - `slog-007` refused new logins on 2026-09-25 when its whole memory pool was
    full. Processes already running there were unaffected.
- **The login node kills your background processes.**
  `/usr/local/sbin/shared-host-watch` SIGKILLs everything left in an ssh
  session's scope about a minute after the session ends. Anything that must
  outlive a command goes in a user unit:
  `systemd-run --user --collect --unit=<name> bash -lc '<cmd>'`. Processes the
  agents start inherit the keeper's unit.
- **Broken Docker on some nodes.** `m2m-341` and `m2m-002` have Docker storage
  that points at deleted directories, and `m2m-042` hung a smoke test. If this
  node is preempted, test a replacement with `docker images` before
  provisioning it.
- **Scripts written from Windows** carry CRLF. A stray `\r` turned a successful
  boot into exit 127.
- **NFS lag.** The login node sees a `STATUS` file the node just wrote on
  `/home` late. Retry, or read it through the node.
- **Containers write only to node-local disk.** `/home` is NFS with
  `root_squash`: write to `/mnt/m2m_nobackup` and copy durable results back.
  `/home` is over 90% full.
- **`pgrep -f` matches itself.** Inside `bash -lc "..."`, `pgrep -f` matches its
  own command line. Use the `[m]easure_one` form.
- **Heredocs.** A heredoc into `docker exec` needs `-i`.
- **Liveness.** A driver that records `STATUS=RUNNING` and then dies looks
  healthy. Check the pid or unit, not the status file.
- **Seed.** The InferenceX mount hardcodes seed 42 (Plan 09 §5).
