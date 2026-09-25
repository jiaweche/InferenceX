# DeepSeek V4.1 Flash Scheduling Bubbles Plan

> Takes over from [Plan 14](./14_DSV41_FLASH_PROFILE_HOST_LAUNCH.md), cancelled
> on 2026-09-25, and works beside
> [Plan 15](./15_DSV41_FLASH_PROFILE_COLLECTIVES_FOLLOWUP.md). The source
> profile, fixed baseline and performance rules are shared and kept once, in
> [Plan 09](./09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md) §0, §3 and §5.
> This is an internal execution document, not published InferenceX documentation.

Status: **submitted** to the Plan 11/13/14 executor mesh on 2026-09-25 (setup in §8)

## 1. Objective

At the load the AgentX benchmark actually runs, account for every millisecond of
a decode step as kernel time or a classified GPU bubble, on all four ranks, at
C8 and C32. Then remove the largest bubble class that no other plan owns.

Plan 14 asked whether HIP runtime calls are exposed. This plan asks what the GPU
is waiting for, whatever the cause, including causes that involve no host call.

## 2. Evidence

### 2.1 Low-batch decode steps look idle

From the matched-batch comparison in `agent/HANDOFF-dsv41-trace-analysis.md`
§4.7 (jiaweichen-amd), on Plan 09's knob-off C8 capture of this base:

| per decode step | B200, 1 req | MI355X, 1 req | B200, 2 req | MI355X, 2 req |
| --- | ---: | ---: | ---: | ---: |
| GPU span, first kernel to last | 7.37 ms | 11.66 ms | 8.19 ms | 12.29 ms |
| sum of kernel time | 8.51 ms | 9.49 ms | 9.34 ms | 10.94 ms |
| span minus kernel time | −1.14 ms | **+2.17 ms** | −1.15 ms | **+1.35 ms** |

- B200's negative figure is overlap: it runs about 1.1 ms of kernels
  concurrently on other streams. MI355X runs nothing concurrently.
- At 7–12 requests (the Ruby C32 deep trace, 45 minutes into its replay) span
  and kernel time agree, 17.88 against 17.87 ms.
- **Treat the MI355X idle figures as an upper bound.** §4.7 excluded DSpark's
  kernels from the kernel sum because they launch outside the annotated steps
  (10–15% of kernel time). Any of them that execute inside a target step's span
  are counted as idle. §2.2 measured about half as much steady idle. §4.1
  settles which is right.

### 2.2 What Plan 14 measured before it was cancelled

Plan 14's `t02` and `t03`, on Plan 10's rank-0 captures of this base
(`14_host_launch/t02/T02_C8_EXPOSED_HOST_TIME.md`, `t03/T03_GRAPH_LAUNCH_INVENTORY.md`):

- **No capture in the campaign ran at its nominal concurrency.** Across 37
  captures, the "C8" ones averaged 0.62–1.05 running requests (maximum 5) and
  the "C32" ones 0.95–1.89 (maximum 10). 32 of 37 opened on `capture_one.sh`'s
  420 s timeout because the occupancy trigger never fired. Every MI355X figure
  labelled C8 or C32 so far, §2.1's included, describes about 0.7 and 1.5
  running requests.
- **At mean occupancy 0.62:**

  | quantity | ms/step | share of an 18.79 ms step |
  | --- | ---: | ---: |
  | raw GPU idle | 2.46 | 13.1% |
  | of which code-object loading (eight `hipModuleLoadDataEx`, steps 240–272) | 1.38 | 7.3% |
  | steady GPU idle, loading removed | 1.09 | 5.8% |
  | of which gaps averaging 0.87 µs, about 1,190 per step, host blocked in `hipEventSynchronize` | 1.04 | 5.5% |
  | idle in gaps of 50 µs or more, all causes | 0.026 | 0.14% |
  | idle exposed by `hipGraphLaunch` | 0.002–0.33 | ≤ 1.7% |

  So at that load the steady idle is device-side gaps between kernels that
  were already queued, not the host failing to keep up. The one large item was
  code loading, which a warmup might or might not have paid.
- **Breakable graphs barely multiply launches**: 2.19 graph launches per step
  on the "C8" capture and 2.46 on "C32".
- **Two method errors to avoid.** "Which host call is in flight" picks a
  blocking wait that spans the step, so it names the waiter, not the cause. And
  summing API durations double-counts across host threads: 21.03 ms of "host
  time" in an 18.79 ms step. Plan 14 §2's 6.93 ms per step of host time is
  therefore an upper bound, not an occupancy.

### 2.3 What neither measurement covers

- **Between steps.** The gap from step N's last kernel to step N+1's first:
  scheduler output, input preparation, sampling and output processing.
- **The DSpark draft.** How draft and verify steps hand off, and what the GPU
  does between them.
- **Waits that look like kernel time.** A custom allreduce spins until every
  rank arrives, so a late peer looks like a long collective. Plan 15 §2.3 finds
  1.90 ms per step of the "C8" decode allreduce (70.7% of it) unexplained by
  kernel work. Plan 15 §4.0 is measuring arrival spread on all four ranks.
- **The benchmark's own load.** The measured replay is 600 s after a discarded
  warmup, with sessions idling between turns, so its running-request count is
  far below the nominal concurrency. It has never been recorded. That
  distribution, not the nominal C8 or C32, decides which bubbles matter.

### 2.4 Who owns what

| bubble class | owner |
| --- | --- |
| host time inside HIP runtime calls (graph launch, stream-capture queries, launches, `hipPointerGetAttribute`) | Plan 14's scope. Plan 14 is cancelled and someone else is taking that scope up. This plan measures and reports the class but builds no runtime fix. |
| collective kernel time, and cross-rank arrival spread | Plan 15 |
| device-side gaps between queued kernels | this plan (§4.3 F) |
| code-object loading inside the measured replay | this plan (§4.3 G) |
| host code inside a step, eager graph breaks, between-step gaps, sync round trips, the draft handoff | this plan |
| why the late rank is late | this plan, using Plan 15's spread |
| missing multi-stream overlap | this plan, last (§4.3 E) |

This plan changes no kernels and no collective selection.

## 3. Fixed baseline

As Plan 09 §3, including `VLLM_USE_BREAKABLE_CUDAGRAPH=1` from the env file.
Only host-side scheduling, synchronization, graph partitioning, stream
assignment, code-object loading, and runtime or graph dispatch settings may
change.

## 4. Work items

### 4.1 Gap census at the benchmark's load

**Operating point.** Record the running-request count through one measured
replay at C8 and one at C32, each after its discarded warmup, exactly as Plan
09's `measure_one.sh` runs them. Sample the server's `/metrics`
(`vllm:num_requests_running`) every few seconds. This is the distribution the
census must represent.

**Captures.** For each point, three windows of 400 `ProfilerStep#`:
- open them at fixed times into the measured replay, for example 120 s, 300 s
  and 540 s after the warmup. The profiler collects once per server lifetime, so
  each window needs a fresh server;
- record all four ranks, with CPU activity;
- record running requests for every step in the window;
- **never open a window on a timeout fallback.** A window records the occupancy
  it actually had, and the census is weighted by §4.1's distribution.

If Plan 15's four-rank captures of this base exist, analyse them first as a
rehearsal, labelled with the occupancy they had.

**Gaps.** Count every interval with no kernel running on any stream of the
device, **at any length**. Report them in length buckets: < 1 µs, 1–10 µs,
10–50 µs, 50–500 µs and ≥ 500 µs. Different causes live at different lengths,
and a floor would discard the sub-microsecond class that is 95% of the idle
Plan 14 measured.

**Classify by whether work was queued**, not by which call was in flight:

- **The next kernel was already submitted when the gap began.** Its launch call
  had returned, or it belongs to a graph replay already launched. The gap is
  device-side.
  - `inter_kernel`: the gap between queued kernels.
  - `jit_load`: the device waited on code-object loading or compilation.
- **The next kernel was not yet submitted.** The host is the cause. Classify by
  what the host thread did between the previous kernel's end and that launch:

  | class | the host thread was… |
  | --- | --- |
  | `hip_runtime` | inside a runtime call. Split `hipGraphLaunch` into back-pressure and overhead, booking only the overlap with idle, as Plan 14 `t02` did. |
  | `host_in_step` | in code between launches inside a step. Name the innermost annotation. |
  | `graph_break` | in the eager segment between two graph launches of one step. Name the break. |
  | `between_steps` | outside the step annotation. Split into waiting for the scheduler's input and the worker's own pre- and post-processing. |
  | `sync_roundtrip` | resuming after a blocking synchronize or device-to-host copy returned. Name the call site. |
  | `unattributed` | none of the above. |

- **`draft` tag.** Mark any gap adjacent to a draft-model launch as `draft`,
  whatever its class.
- **Wall clock only.** Measure each host thread on its own timeline and never
  sum API durations.

**Hidden waits.** Align the four ranks' decode collectives by step and layer,
with Plan 15's arrival spread or the same method. For each collective whose
spread exceeds its kernel duration, record the late rank and its class just
before its launch.

**Report**, per rank and per point, bucketed by running requests in the step
(1, 2, 3–6, 7–12), and weighted by §4.1's distribution:

- step wall time = kernel busy time (the union over streams) + Σ classes, in
  ms per step;
- the hidden-wait total, which sits inside kernel busy time, beside it;
- gap counts and time per length bucket;
- the top five call sites or breaks in each host class;
- whether `jit_load` occurs after the warmup, i.e. inside what the benchmark
  measures.

**Profiler tax.** Kineto's CPU tracing inflates exactly the host classes; Plan
10 measured its own decode probe at +21%. Compare time per output token in each
profiled window with the same point of an unprofiled replay. Report the host
classes as upper bounds if the profiled window is more than 5% slower.

### 4.2 Explain the largest classes

Before designing a fix, find the mechanism of the two largest classes this plan
owns, at each point:

- **`inter_kernel`:** kernels per step and gap per kernel, inside graph replays
  against eager launches. Run a microbenchmark of back-to-back trivial kernels,
  in a graph and on a stream, on idle GPUs 4–7. That gives this image's
  intrinsic dispatch gap to compare with production.
- **`jit_load`:** which code objects load, and when. Would a longer warmup, or
  loading at startup, have paid for it before the measured replay?
- **`between_steps`:** time the scheduler per step in the engine core process,
  which the worker traces cannot see. Time the worker's input preparation and
  output processing. Confirm whether the control server runs vLLM's async
  scheduling, and whether this build supports it together with DSpark.
- **`sync_roundtrip`:** list every blocking synchronize and device-to-host
  copy on the decode path, with what consumes its value and whether that is
  needed before the next launch.
- **`graph_break`:** name the op that forces each break, and give each break's
  cost.
- **`host_in_step`:** sample one worker with `py-spy` during an unprofiled
  replay, for Python frames without Kineto's cost.
- **Late rank:** is it always the same rank, and what does it do that the
  others do not?

### 4.3 Candidates, one knob each, default off

Build only for classes that §4.1 prices at 0.5 ms per step or more at C8 or
C32, largest first. Each candidate names its class and a predicted saving
before it is measured.

- **F. Shorter dispatch between queued kernels** (`inter_kernel`): runtime or
  graph settings that ROCm documents as reducing per-kernel dispatch latency.
  Cite each one, prove it on §4.2's microbenchmark first, and knob it. If none
  moves the microbenchmark, report kernels per step and the per-kernel gap to
  the kernel plans, since fusion is then the lever.
- **G. No code loading in the measured replay** (`jit_load`): load eagerly at
  startup, or warm the missing kernels in the warmup. Only if §4.1 finds
  loading after the warmup.
- **A. Overlap scheduling with execution** (`between_steps`): vLLM's async
  scheduling if §4.2 finds it supported with DSpark; otherwise the narrowest
  change that prepares step N+1's inputs while step N runs.
- **B. Remove avoidable syncs** (`sync_roundtrip`): make each unneeded blocking
  copy non-blocking or deferred, one call site at a time, each named with its
  file and line.
- **C. Fewer decode graph breaks** (`graph_break`): capture the breaking op, or
  move the break off the decode path.
- **D. Balance the late rank** (hidden waits): move host work that only the
  late rank does off the path to the step's first collective.
- **E. Multi-stream overlap** (the idle B200 fills by overlapping): run two
  independent pieces of work concurrently. Last and largest. It needs a
  data-dependency proof first; timing overlap alone does not show independence
  (Plan 11's review).

### 4.4 Correctness

A scheduling change must not change what is generated.

- Greedy generation of a fixed prompt set must be token-identical between the
  arms.
- A change to spec-decode control flow (the draft, acceptance or verify timing)
  also needs full-length GSM8K with real DSpark block rejection, for both arms.
- A change to stream assignment, graph capture or dispatch settings also needs a
  graph-replay soak with no hang, before any timed run. IPC buffer registration
  during capture is the known hazard (Plan 14 §4.4).

## 5. Gates

- **Operating point recorded.** Every capture window carries the running
  requests it had, and the census is weighted by the measured replay's
  distribution.
- **Census closes.** On every rank and at both points, kernel busy time plus the
  classes equals step wall time within 1%, and `unattributed` is under 5% of
  idle. Otherwise fix the instrument before designing any candidate.
- **Trace.** At matched occupancy buckets, the targeted class falls, **total**
  idle per step falls, and step wall time falls. A bubble that moves to another
  class is not a result.
- **Correctness.** §4.4, before any performance run.
- **Performance.** Plan 09 §5. Report C8 first, then C32. A C32 loss makes the
  candidate a Pareto point, not a win.

## 6. Stop conditions

- The census finds less than 0.5 ms per step of bubbles in classes this plan
  owns, at both C8 and C32.
- The largest class belongs to Plan 15, or to Plan 14's scope. Hand over the
  census and stop.
- The largest class is `inter_kernel`, and nothing moves §4.2's
  microbenchmark. Report kernels per step and the per-kernel gap to the kernel
  plans, and stop.
- The profiler tax exceeds the host classes it measures, and sampling cannot
  resolve them without it.
- A candidate changes generated tokens, fails GSM8K, or hangs under graph
  replay.

## 7. Artifacts

```text
crusoe:/home/jiaweche/dsv41-profile-20260924/16_scheduling_bubbles/
```

Persist:

- the running-request distributions of the measured replays;
- the classifier, and its per-rank census tables by occupancy and gap length;
- the hidden-wait alignment and the profiler-tax comparison;
- the microbenchmark with its raw samples, and the `py-spy` samples;
- each candidate's knob diff with its hash, and the correctness outputs;
- the captures (raw traces on node-local disk, summaries on `/home`);
- per-replay bench outputs, and the decision.

## 8. Execution setup

Written by the operator before submission. These are facts about the
environment, not choices about the plan.

### 8.1 Mesh, node and trees

- **Mesh.** This plan runs on the mesh that executed Plans 11, 13 and 14:
  - Session: `/home/jiaweche/Arbor-plan-executor-p11/sessions/plan-executor`.
  - Config: `configs/mi355x/plan-executor-p11.json`.
  - Keeper: user unit `arbor-p11-keeper` on `slog-007`.
  - Plans 11 and 13 settled with stop and their reviews are done. Plan 14 was
    cancelled by the operator (board plan `p03`, abandoned) with no review
    requested.
- **Node.** Job `174831` on `crsuse2-m2m-016` (`amd-burst`, 24 h, preemptible).
  The pool adopts jobs named `arbor-p11-*`. All four of the account's job slots
  are in use.
- **Container.** `dsv41flash_arbor` was recreated from the image by digest
  (`sha256:960228cf…`) at 20:46 UTC on 2026-09-25, with the fixed base
  `be794db46` overlaid. Checked at 21:30 UTC:
  - no GPU process on the node, and no server in the container;
  - `.deployed_sha` reads `be794db469808b5611ee2f6f85207a59bc807931`;
  - AITER is stock 0.1.21.post2, and its JIT caches were cold at recreation;
  - weights are node-local at
    `/mnt/m2m_nobackup/jiaweche/inferencex-dsv41flash/models/DeepSeek-V4.1-Flash`;
  - the aiperf client is at `/runtime/aiperf-src`.

  **Stop every server you start before a task closes.** Plan 13 left an
  orphaned server holding GPUs 0–3.
- **Other tenants' containers** (`miles_primus_yuankai`,
  `oshkarav-inference-testing-amd-1`) sit on this node. Neither held a GPU at
  setup. Record GPU occupancy before every measured stage.
- **Spur runs job steps in their own PID namespace.** Host PIDs from
  `amd-smi process` are not in `/proc` inside `srun --overlap`. Map them with
  `docker top <container> -eo pid,lstart,args`.
- **Not ours.** Never touch these jobs, their containers, or their sessions.
  Read-only access to their files is fine.

  | Job | Node | Belongs to | Mesh runs on | Session |
  | --- | --- | --- | --- | --- |
  | `174800` | `m2m-032` | Plan 10 | `slog-005` | `/home/jiaweche/Arbor-plan-executor` |
  | `174801` | `m2m-191` | Plan 12 | `slog-006` | `/home/jiaweche/Arbor-plan-executor-p12` |
  | `174972` | `m2m-193` | Plan 15 | `slog-006` | `/home/jiaweche/Arbor-plan-executor-p15` |

- **Paths.**
  - Artifacts (§7): `R=/home/jiaweche/dsv41-profile-20260924/16_scheduling_bubbles`.
  - Base checkout: `/home/jiaweche/dsv41-merge/vllm-v6best-pr19`, a worktree of
    `/home/jiaweche/dsv41-merge/vllm`. Do not commit on it.
  - This plan's branch: a new worktree from `be794db46`, for example branch
    `arbor-v2-plus-v6best-pr19-p16` at `/home/jiaweche/dsv41-merge/vllm-p16`.
- **Deploying Python changes.** Run
  `SRC=<worktree> WANT_SHA=<full sha> bash $R/bin/deploy_tree.sh` on the node.
  It overlays `vllm/`, hash-checks every `.py` in the commit against the
  container, confirms the compiled extensions are untouched, and imports the
  DSA attention modules. Anything short of `DEPLOY_OK sha=<sha>` means the tree
  is not what runs.
- **Server configuration.** The control server's env file, which supplies §3's
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, is `server_env_blind.list` in the v9 toolbox
  (§8.2), md5 `64a08abe4e522e6ba451e4b6bebdebcb`.
- **MoE configs.** The base's PR #19 MoE-config install writes its CSV into the
  AITER package inside the container, so every server after the first runs the
  FlyDSL MoE configs. That is identical in both arms of every A/B here and must
  stay so. Plan 09 owns it, and found it costs about 3.5% at C32.
- **GPUs.** A server runs TP4 on GPUs 0–3. GPUs 4–7 are idle, which is where
  §4.2's microbenchmark runs without disturbing a server.
- **Other plans.** Plans 09–15 are in `/home/jiaweche/dsv41-profile-20260924/plans/`.

### 8.2 Existing work to reuse

- **Plan 14's partial results**, in
  `/home/jiaweche/dsv41-profile-20260924/14_host_launch/`:
  - `t02/`: the gap attribution at occupancy 0.62, with `gap_attrib.py`,
    `gapdist.py`, `jitsplit.py`, raw per-gap rows, and `OCCUPANCY_CENSUS.tsv`
    covering 37 captures;
  - `t03/`: the graph-launch inventory and `exposure_bound.py`.

  §4.1's rule of classifying by whether work was queued replaces `t02`'s
  in-flight rule. Its gap rows are still useful as a first pass.
- **Plan 15** runs on its own mesh (`slog-006`). Its §4.0 is capturing the base
  on all four ranks to measure allreduce arrival spread. Its artifacts are in
  `/home/jiaweche/dsv41-profile-20260924/15_collectives_followup/`: read,
  never write.
- **Rank-0 captures of this base**, all at 0.6–1.9 running requests
  (`OCCUPANCY_CENSUS.tsv`):
  - Plan 10: `.../10_dsa_indexer_decode/run1/cap_base_c{32,8}/traces/`;
  - Plan 09: `.../09_moe_experts/run1/trace_a{0,1}_c{8,32}/traces/`.
- **Scripts.** Copy and change what differs:
  - Plan 09 (`.../09_moe_experts/run1/bin/`): `measure_one.sh` (one A/B point:
    fresh control server, marker check, discarded warmup, one measured replay),
    `capture_one.sh`, `p09_driver.sh` and `summarize_ab.py`.
    **`capture_one.sh`'s timeout fallback is the defect in §2.2.** Replace it
    with windows opened at fixed times into the measured replay.
  - Plan 10 (`.../10_dsa_indexer_decode/run1/bin/`): `dsa_attrib.py` (Kineto
    attribution, including graph-replayed kernels via the `correlation` of their
    `hipGraphLaunch`) and `t01_driver.sh` (captures under a user unit).
- **Harness.** Read, never edit; it belongs to another mesh's session:
  `/home/jiaweche/Arbor-dsv41-mesh/sessions/megamoe-v9/mesh/toolbox/crusoe/{boot_control_server.sh,replay_arm.sh}`.

### 8.3 Things that have already cost time here

- **What Plan 11's review found.** Avoid repeating these:
  - **A gate that was not run is "not run"**, not "passed", even after a stop
    makes it moot.
  - **Keep the raw samples** of each microbenchmark replay, and record the
    exact script that produced them.
  - **Step counts come from `ProfilerStep#`.** Never infer them from latency.
  - **Timing overlap is not independence.**
- **When a settled plan's work is done, it is done.** After Plan 13 settled,
  this mesh's agents spent more than an hour messaging one another about its
  records. Plan 14's leftover mail is in some inboxes: it was cancelled, so do
  not act on it.
- **Memory.** The keeper's unit is capped at 2200M with `OOMPolicy=continue`,
  and the login node caps the user at 4 GiB.
  - Parse traces on the node through `srun --overlap --jobid=174831`, never on
    the login node.
  - An agent that parsed a Kineto trace on a login node grew to 1.8 GB and
    slowed every process of this user on that host.
  - `slog-007` refused new logins on 2026-09-25 when its memory pool was full.
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
  `root_squash` and 93% full: write to `/mnt/m2m_nobackup` and copy durable
  results back.
- **`pgrep -f` matches itself.** Inside `bash -lc "..."`, `pgrep -f` matches its
  own command line. Use the `[m]easure_one` form.
- **Heredocs.** A heredoc into `docker exec` needs `-i`.
- **Liveness.** A driver that records `STATUS=RUNNING` and then dies looks
  healthy. Check the pid or unit, not the status file.
- **Seed.** The InferenceX mount hardcodes seed 42 (Plan 09 §5).
