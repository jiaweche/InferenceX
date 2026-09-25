# DeepSeek V4.1 Flash TP Allreduce Plan

> One of six profile-driven plans (09–14) from the MI355X C32 deep trace of
> 2026-09-24. The source profile, fixed baseline and performance rules are
> shared and kept once, in [Plan 09](./09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md)
> §0, §3 and §5.
> This is an internal execution document, not published InferenceX documentation.

Status: **settled, stop** — 2026-09-25 13:48 UTC by its plan-executor mesh
(setup in §8). Settlement:
`crusoe:/home/jiaweche/dsv41-profile-20260924/11_tp_allreduce/T09_SETTLEMENT.md`.
Successor: [Plan 15](./15_DSV41_FLASH_PROFILE_COLLECTIVES_FOLLOWUP.md)

> **Base changed (2026-09-24)** to `arbor-v2-plus-v6best-pr19` (Plan 09 §3).
> The code references below (fusion enablement, the one-stage cap) were read at
> `93205f244`. Recheck them on the new base before execution.

> **Premise corrected (2026-09-25).** The 3.8× B200 gap below compared
> different batch sizes. At matched batch the decode allreduce is 1.6–1.7×
> slower, and decode runs far below the 48 and 192 rows assumed in §2.1: 6–12
> rows in a shallow C8 window and 42–72 in the deep C32 trace. Details in §2 and in
> `agent/HANDOFF-dsv41-trace-analysis.md` §4.7 (jiaweichen-amd). The mesh
> executing this plan works from its pinned copy and does not see this edit.

## 1. Objective

Lower the per-call latency of the TP4 allreduce, above all the decode
allreduce. It is 15% of GPU kernel time, all on the critical path, and none of
the open kernel work touches it. At matched batch it is 1.6–1.7× slower than
B200's; the gap at C32's steady-state batch is not yet known.

## 2. Evidence

Collectives are **15.08%** of GPU kernel time, all of it on the critical path:

| kernel | population | calls | mean µs | p50 µs | ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `aiter::cross_device_reduce_2stage<bf16,4>` | decode | 33,650 | 21.1 | 19.5 | 710.44 |
| | prefill | 1,558 | 460.9 | | 718.14 |
| `ncclDevKernel_Generic_1` (RCCL) | decode | 1,594 | 59.8 | 69.1 | 95.31 |
| | prefill | 92 | 1638.9 | | 150.77 |
| **total** | | **36,894** | | | **1674.65** |

- **The gap is per call, and it grows with batch.** B200's C32 share is about
  the same, 15.93%. Its decode allreduce (flashinfer `trtllm_mnnvl
  oneshotAllreduceFusionKernel`) has a p50 of 5.1 µs, but only over steps of
  1–2 requests, which is all the B200 traces contain. The 19.5 µs here is over
  steps of 7–12 requests. Matched by batch, on decode-only steps with 81
  allreduces each:

  | requests per step | rows | B200 p50 | MI355X p50 | MI355X kernel |
  | ---: | ---: | ---: | ---: | --- |
  | 1 | 6 | 4.7 µs | 7.4 µs | vLLM one-stage |
  | 2 | 12 | 5.6 µs | 9.5 µs | AITER one-stage |
  | 8 | 48 | not profiled | 14.2 µs | AITER two-stage |
  | 10 | 60 | not profiled | 19.8 µs | AITER two-stage |

  MI355X at 1–2 requests is Plan 09's knob-off C8 capture; at 8–10 requests it
  is the deep C32 trace. So the gap is 1.57× and 1.70× where it can be
  measured, and unknown at the batch C32 actually runs.
- **Nothing overlaps.** `total_comm_time` equals `exposed_comm_time` to within
  0.06%, and the custom allreduce runs inline on the compute stream.
- **`cross_device_reduce_1stage` never appears** in the deep run, whose steps
  are all 7 requests or more. It carries every decode allreduce in the shallow
  C8 capture, at 1–2 requests.
- **Split the population before quoting a per-call cost.** The kernel is
  bimodal: decode and prefill differ by about 20×. An earlier "7× gap" came from
  a mean over both populations and was wrong.
- **Read TraceLens comm figures with care.** TraceLens classifies
  communication by kernel name. It files the AITER custom allreduce as
  computation (the 719.6 ms `vllm::all_reduce` of the shallow trace is four
  times its whole comm bucket), so its "2.20% exposed" is not an overlap
  figure.
- **The previous tree measured it higher.** Arbor v8 p10-t04 put
  `tp_all_reduce` at 26.0% of GPU-busy on that tree.

### 2.1 Why decode takes the two-stage kernel

At `93205f244`:

- AITER custom allreduce is on by default (`VLLM_ROCM_USE_AITER_CUSTOM_AR=True`).
- At optimization levels O2 and O3, the allreduce + RMSNorm fusion pass is
  enabled on ROCm whenever AITER is on and TP > 1 (`enable_allreduce_rms_fusion`,
  `vllm/config/vllm.py:171`). Whether this server's decode allreduces actually
  go through the fused op is step 4.1's first question.
- `AiterCustomAllreduce.use_1stage_fused_ar_rms`
  (`vllm/distributed/device_communicators/aiter_custom_all_reduce.py`) admits
  the one-stage kernel only for at most 80 tokens **and** under 256 KB at TP ≤ 4.
  At hidden 5120 in bf16 a row is 10 KiB, so the cap is 25 rows.
- 48 rows (C8) and 192 rows (C32) are the per-step **maxima**,
  concurrency × (1 + 5 speculative tokens). Measured steps carry far fewer
  requests: the shallow C8 capture runs 1–2 per step (6–12 rows, inside the
  cap), and the deep C32 trace runs 7–12 (42–72 rows, 420–720 KiB). So the
  one-stage cap binds at C32. In the C8 window measured so far one-stage
  already runs; steady-state C8 crosses the 25-row cap only at 5 or more
  requests per step, which has not been measured. The two-stage variant's own
  docstring says it "is slower than an explicit `all_reduce` + norm".

The 256 KB cap mirrors AITER's launcher contract
(`csrc/include/custom_all_reduce.cuh`), so raising it may be an AITER change,
not a vLLM constant.

### 2.2 The other populations

- **Prefill custom allreduce** (1,558 × 460.9 µs, 6.47%) is as large as decode
  in total.
- **RCCL decode calls** (1,594 × ~60–69 µs) are not on the custom path at all.
  The shallow report's `coll_analysis` matches only RCCL `_allgather_base`.
- **QuickReduce** (`vllm/distributed/device_communicators/quick_all_reduce.py`,
  `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`) exists in the tree. It targets large
  messages with quantized transport.

## 3. Fixed baseline

As Plan 09 §3. Only collective selection and collective kernel parameters may
change.

## 4. Work items

### 4.1 Attribute every collective to its call site

For every collective kernel in the deep trace and the shared C8 capture
(Plan 09 §4.1), record:

- the calling op: plain `all_reduce`, fused AR + RMSNorm (+ per-group quant),
  or all-gather;
- bytes per call;
- the path taken: one-stage, two-stage or RCCL.

Name the op behind the 1,594 RCCL decode calls.

### 4.2 Find the real crossover on MI355X

On one TP4 node, with graph capture, bench all four options against each other:

- AITER one-stage;
- AITER two-stage;
- fused AR + RMSNorm in both variants;
- RCCL.

Sweep hidden 5120 bf16 over 1–256 rows (decode) and 1K–16K rows (prefill).
Report p50 and p99 per size. The question is whether one-stage beats two-stage
at 42–72 rows, the range C32 decode actually runs, and by how much. Report 48
and 192 rows too, as the per-step maxima.

### 4.3 Candidates, one knob each, default off

- **A. One-stage decode** up to the crossover measured in 4.2, for plain and
  fused allreduce. If the cap lives in AITER's launcher, carry the AITER change
  as a pinned patch and record its hash.
- **B. Route the RCCL decode stragglers** to AITER's custom path
  (`should_custom_ag` / `custom_all_gather`) where 4.1 shows they fit.
- **C. QuickReduce for prefill-size messages.** This changes numerics, so it
  needs the accuracy gate.
- **D. Overlap.** In scope only if 4.1 finds independent work adjacent to a
  collective. Not assumed.

### 4.4 Accuracy

A and C change reduction order or transport precision. Compare against RCCL
with a per-element tolerance, then run the 1,319-example GSM8K with real DSpark
block rejection for both arms.

## 5. Gates

Kernel: A must beat the two-stage p50 across 42–72 rows (14.2 µs at 48 rows and
19.8 µs at 60 in the deep trace). In the measured C8 window decode already runs
one-stage, so A can move C8 only if steady-state C8 steps reach 5 or more
requests.

Trace, on a post-change capture:

- the one-stage kernel appears for decode rows inside the new cap;
- decode allreduce time per profiler step falls below the baseline 1.78 ms/step
  (710.44 ms / 400 `ProfilerStep#`);
- no new RCCL fallbacks appear.

Performance: Plan 09 §5. Both points matter here. C8 and C32 cross the
one-stage cap at different row counts, so a cap tuned at one says nothing about
the other.

## 6. Stop conditions

- 4.2 shows two-stage already optimal at 48 and 192 rows. Then decode latency is
  a transport limit, and the remaining lever is overlap (D) or fewer bytes (C).
- An AITER change is needed and cannot be pinned reproducibly in the image.
- Any hang or corruption appears under graph replay. Stop immediately; IPC
  buffer registration during capture is involved (see Plan 14 §4.4).

## 7. Artifacts

```text
crusoe:/home/jiaweche/dsv41-profile-20260924/11_tp_allreduce/
```

Persist the call-site table, the crossover sweep, any AITER patch with its
hash, the knob diffs, the trace captures, per-replay bench outputs and the
decision.

## 8. Execution setup

Written by the operator before submission. These are facts about the
environment, not choices about the plan.

### 8.1 Node and trees

- **Node.** Job `174710` on `crsuse2-m2m-250` (`amd-burst`, 24 h, preemptible).
  It is this mesh's only node. The pool adopts jobs named `arbor-p11-*` and
  will not request another while it holds this one.
- **Container.** `dsv41flash_arbor` runs the image by digest (`sha256:960228cf…`).
  Weights are node-local at
  `/mnt/m2m_nobackup/jiaweche/inferencex-dsv41flash/models/DeepSeek-V4.1-Flash`,
  and the aiperf client is at `/runtime/aiperf-src`.
- **Base on the node.** The fixed base `be794db46` is overlaid on the container.
  The setup log is `$R/logs/setup.log`.
- **Not ours.**
  - Job `174441` on `m2m-016` is Plan 09's A/B.
  - Job `174682` on `m2m-031` is Plan 10's node.
  - Plan 10's mesh runs on `slog-005` from `/home/jiaweche/Arbor-plan-executor`.
  - Never touch these jobs, their containers, or that session. Read-only
    access to Plan 10's files is fine.
- **Paths.**
  - Artifacts (§7): `R=/home/jiaweche/dsv41-profile-20260924/11_tp_allreduce`.
  - Base checkout: `/home/jiaweche/dsv41-merge/vllm-v6best-pr19`, a worktree of
    `/home/jiaweche/dsv41-merge/vllm`. Do not commit on it.
  - This plan's branch: make a new worktree from `be794db46`, for example
    branch `arbor-v2-plus-v6best-pr19-p11` at
    `/home/jiaweche/dsv41-merge/vllm-p11`. `vllm-p10` and `vllm-p10-instr`
    belong to Plan 10.
- **Deploying Python changes.** Run
  `SRC=<worktree> WANT_SHA=<full sha> bash $R/bin/deploy_tree.sh` on the node.
  It overlays `vllm/`, hash-checks every `.py` in the commit against the
  container, confirms the compiled extensions are untouched, and imports the
  DSA attention modules. Anything short of `DEPLOY_OK sha=<sha>` means the tree
  is not what runs.
- **Deploying compiled changes.**
  - vLLM: Plan 10 built a hash-checked rebuild-and-install path for its
    compiled extension, in
    `/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/bin/`
    (`build_so.sh`, `deploy_so.sh`).
  - AITER: its custom allreduce lives in the image's own `aiter` package
    (AITER 0.1.21.post2), not in vLLM, so an AITER change needs its own path.
    Carry it as a pinned patch with its hash (§4.3 A), and prove the rebuilt
    module is the one the server loads.
- **Other plans.** Plan 09 (§0, §3, §5 above) and Plan 14 (cited in §6) are
  `/home/jiaweche/dsv41-profile-20260924/plans/09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md`
  and `.../plans/14_DSV41_FLASH_PROFILE_HOST_LAUNCH.md`.
- **MoE configs.** The base's PR #19 MoE-config install writes its CSV into the
  AITER package inside the container, so every server after the first runs the
  FlyDSL MoE configs. That is identical in both arms of every A/B here and must
  stay so. Plan 09 owns it.
- **GPUs.** The server runs TP4 on GPUs 0–3. GPUs 4–7 are idle.

### 8.2 Existing work to reuse

- **Traces of this base, readable now.**
  - Plan 10's fresh knob-off captures of `be794db46` at C32 and C8, 400
    `ProfilerStep#` each, taken on `m2m-031` with the image's stock compiled
    extension. They are in
    `/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/cap_base_c32/traces/`
    and `cap_base_c8/traces/`, summarised in `run1/T01_BASELINE.json`.
  - Plan 09's captures of `cbfdad177`, with the MoE-config knob at 0 and at 1,
    at C32 and C8:
    `/home/jiaweche/dsv41-profile-20260924/09_moe_experts/run1/trace_a{0,1}_c{8,32}/traces/`.
  - Only the **rank-0** trace of each capture is on `/home`. A collective's
    time on rank 0 includes waiting for the other three ranks, which rank 0
    alone cannot separate. Anything compared against a candidate is captured
    on this node.
- **Scripts.** Copy and change what differs:
  - Plan 09 (`/home/jiaweche/dsv41-profile-20260924/09_moe_experts/run1/bin/`):
    `capture_one.sh`, `measure_one.sh` (one A/B point: fresh control server,
    marker check, discarded warmup, one measured replay), `eval_one.sh`
    (GSM8K with real block rejection), `p09_driver.sh` and `summarize_ab.py`.
  - Plan 10 (`/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/bin/`):
    `dsa_attrib.py` (Kineto attribution, including graph-replayed kernels),
    `t01_driver.sh` (captures under a user unit), knob probes, `build_so.sh`
    and `deploy_so.sh`, and `t04_bench.sh` (a standalone kernel bench).
- **Plan 10's decisions on shared questions.** These include capture-window
  validity and what "knob-off" means for a candidate already in the base. They
  are in
  `/home/jiaweche/Arbor-plan-executor/sessions/plan-executor/mesh/intake/10_DSV41_FLASH_PROFILE_DSA_INDEXER_DECODE.decisions.jsonl`.
- **Harness.** Read, never edit; it belongs to another mesh's session:
  `/home/jiaweche/Arbor-dsv41-mesh/sessions/megamoe-v9/mesh/toolbox/crusoe/{boot_control_server.sh,replay_arm.sh}`.
- **The Ruby deep trace** behind §2 is on `ruby-1`, which Crusoe cannot reach,
  so §2's numbers are the reference, not a file.

### 8.3 Things that have already cost time here

- **The login node kills your background processes.**
  `/usr/local/sbin/shared-host-watch` SIGKILLs everything left in an ssh
  session's scope about a minute after the session ends, `setsid`/`nohup`
  children included, and ends every session at 8 hours. Anything that must
  outlive a command goes in a user unit:
  `systemd-run --user --collect --unit=<name> bash -lc '<cmd>'`. This mesh runs
  on `slog-007`, where lingering was enabled on 2026-09-25, so user units
  survive there. Processes the agents start inherit the keeper's unit.
- **Memory on the login node.** It caps the whole user at 4 GiB per login node,
  with throttling from 2.5 GiB, shared with all five agents of this mesh. Parse
  traces and run anything heavy on the node through
  `srun --overlap --jobid=174710`, never on the login node.
- **Broken Docker on some nodes.**
  - On `m2m-341` and `m2m-002`, `/var/lib/docker` and `/var/lib/containerd` are
    symlinks into `/mnt/m2m_nobackup` whose targets no longer exist, so every
    image operation fails.
  - `m2m-042` hung a three-minute smoke test.
  - If this node is preempted, test a replacement with `docker images` before
    provisioning it.
- **Stale GPU samples.** The mesh's own GPU sampling (`srun` into the node)
  fails intermittently, so recorded node state can lag by minutes.
- **Scripts written from Windows** carry CRLF. A stray `\r` turned a successful
  boot into exit 127.
- **NFS lag.** The login node sees a `STATUS` file the node just wrote on
  `/home` late. Retry, or read it through the node.
- **Containers write only to node-local disk.** `/home` is NFS with
  `root_squash`: write to `/mnt/m2m_nobackup` and copy durable results back.
  `/home` is 91.5% full.
- **`pgrep -f` matches itself.** Inside `bash -lc "..."`, `pgrep -f` matches its
  own command line. Use the `[m]easure_one` form.
- **Heredocs.** A heredoc into `docker exec` needs `-i`.
- **Liveness.** A driver that records `STATUS=RUNNING` and then dies looks
  healthy. Check the pid or unit, not the status file.
- **Seed.** The InferenceX mount hardcodes seed 42 (Plan 09 §5).
