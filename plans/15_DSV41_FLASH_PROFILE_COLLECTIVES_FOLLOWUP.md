# DeepSeek V4.1 Flash Collectives Follow-up Plan

> Successor to [Plan 11](./11_DSV41_FLASH_PROFILE_TP_ALLREDUCE.md), built from
> the directions its settlement filed. The source profile, fixed baseline and
> performance rules are shared and kept once, in
> [Plan 09](./09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md) §0, §3 and §5.
> This is an internal execution document, not published InferenceX documentation.

Status: **submitted** to its own plan-executor mesh on 2026-09-25 (setup in §8)

## 1. Objective

Take the collective savings Plan 11 found but did not pursue, cheapest first:

1. keep the largest prefill allreduces off RCCL (Plan 11 TODO #4);
2. fuse the decode allreduce with the hyper-connection post-mix that always
   follows it (TODO #3);
3. quantize prefill-size allreduces with QuickReduce, behind the accuracy gate
   (TODO #5).

Before any of them, find out how much of the C8 decode allreduce is kernel work
at all.

## 2. Evidence

All figures are from Plan 11's settlement,
`crusoe:/home/jiaweche/dsv41-profile-20260924/11_tp_allreduce/T09_SETTLEMENT.md`,
and its call-site attribution `T03_CALL_SITES.md`, measured on Plan 10's
knob-off captures of the base `be794db46` (400 `ProfilerStep#` each, rank 0 only).

### 2.1 What Plan 11 settled

Verdict **stop**, on measurements that stand without the B200 comparison:

- Forcing the one-stage allreduce is slower at every size from 1 to 16,384 rows.
  Two-stage is 1.83× faster at 48 rows and 2.01× at 192.
- The fused allreduce + RMSNorm path never runs on this model. The residual is a
  hyper-connection mix, so `aiter::mhc_post_kernel` follows the allreduce in
  34,488 of 34,488 post-collective positions at C32 and the RMSNorm fusion pass
  has nothing to match.
- No collective currently overlaps any other kernel (0 of 36,895 at C32). That
  is what the schedule does, not proof that nothing could overlap; overlap
  stays open and is out of scope here.

### 2.2 The dispatch ladder on this build

| message bytes | decode rows | backend | kernel |
| --- | ---: | --- | --- |
| ≤ 81,920 | ≤ 8 | vLLM custom allreduce | `vllm::cross_device_reduce_1stage` |
| 81,921 – 163,839 | 9 – 15 | AITER | `aiter::cross_device_reduce_1stage` |
| 163,840 – 64 MiB | 16 – 6,553 | AITER | `aiter::cross_device_reduce_2stage` |
| > 64 MiB | ≥ 6,554 | RCCL | `ncclDevKernel_Generic_1` |

The 64 MiB ceiling is `AiterCustomAllreduce.effective_max_size()`, `MAX_SIZE // 2`
in `aiter_custom_all_reduce.py`. vLLM retries its own custom allreduce after AITER
declines, but that one tops out at 8 MiB, so the retry never fires and everything
above 64 MiB goes to RCCL.

### 2.3 Measured cost on the base

| item | C32 | C8 |
| --- | ---: | ---: |
| decode allreduce, ms/step | 0.9001 | 2.6799 |
| of which beyond calls × p50 | 22.0% | **70.7%** |
| decode allreduce p50 per call | 8.2 µs (24–48 rows) | 8.2–9.8 µs (6–18 rows) |
| `mhc_post_kernel`, ms/step | 0.4327 | 0.4094 |
| prefill collectives QuickReduce could take | ~2.33 ms/step | |
| five largest allreduces (81–167 MB) | on RCCL, 9.5–10.3% slower than AITER two-stage | |

- **The C8 remainder is the largest collective cost at C8 and nobody knows what
  it is.** Kernel-like time is nearly the same at both points (0.78 against
  0.70 ms/step), yet C8's decode allreduce costs 1.78 ms/step more, 95% of it in
  the part a median-sized kernel does not explain. For the AITER subset, Plan
  11's idle-node bench confirms the remainder is not kernel execution. That it
  is peer arrival skew is a hypothesis: a custom allreduce waits for every rank,
  and only rank 0 was captured.
- **It may be a host-side problem.** At matched batch on decode-only steps, the
  MI355X GPU sits idle about 2.2 ms per step at 1 request and B200 does not
  (`agent/HANDOFF-dsv41-trace-analysis.md` §4.7, jiaweichen-amd). Rank skew
  from host-side jitter would show up as exactly this kind of allreduce wait.
- **B200 context.** Matched by batch in situ, the MI355X decode allreduce is
  1.57× slower at 6 rows and 1.70× at 12. B200 has not been profiled at C32's
  steady-state batch.

## 3. Fixed baseline

As Plan 09 §3. Only collective selection, collective size limits, QuickReduce
settings and the allreduce/post-mix dispatch may change. Plan branch: a new
worktree from `be794db46`, for example `arbor-v2-plus-v6best-pr19-p15`.

## 4. Work items

### 4.0 Baseline capture on all four ranks

Plan 11 had rank 0 only, which cannot separate an allreduce's own time from
waiting for the other ranks. Capture the knob-off base at C8 and C32, 400
steps each, **all four ranks**, same node as every later candidate.

For each decode allreduce, align the four ranks' launches by step and layer,
and record the arrival spread (last rank's kernel start minus first) against
the kernel's duration. Then answer:

- how much of C8's 1.90 ms/step remainder is arrival spread;
- which rank arrives last, and whether that is stable;
- what the late rank was doing just before: kernels, or host gaps.

If the remainder is arrival spread caused by host gaps, it belongs to Plan 14
(§6). Record the finding and continue with 4.1–4.3, whose savings are real
either way.

### 4.1 Keep the largest allreduces off RCCL (TODO #4)

Raise the 64 MiB ceiling so the >64 MiB prefill allreduces take AITER two-stage.
Plan 11's bench did this with `AITER_CUSTOM_AR_MAX_SIZE` at 256 MiB on a
512 MiB IPC pool. First confirm which setting the **server** honours, since the
ceiling in §2.2 is vLLM's `MAX_SIZE` constant. Then:

- one knob, default off, raising the ceiling;
- record the extra IPC pool memory per rank, and the KV cache capacity the server
  reports with it on and off;
- a numerics check, AITER two-stage against RCCL on the same inputs, compared
  element by element. Plan 11 compared only a scalar maximum deviation, which
  its own review ruled does not establish equivalence (settlement §4).

Expected value is about 0.10 ms/step on prefill steps, below the end-to-end
noise floor, so this item is gated on the trace (§5), not on throughput.

### 4.2 Fuse the allreduce with the hyper-connection post-mix (TODO #3)

AITER in this image already builds `allreduce_mhc_post_split_launcher` and
`allreduce_mhc_post_large_m_kernel` (`fused_ar_mhc_post.cu`). Nothing dispatches
them. In order:

1. **Semantics.** Show that the fused kernel computes what vLLM's `MHCPostOp`
   computes after the allreduce (the post-mix and residual-mix update of the 4
   streams), for the dtypes and layouts this model uses. Compare against the
   unfused pair on real activations, element by element. If it computes
   something else, stop this item (§6).
2. **Kernel bench.** Fused against allreduce + `mhc_post_kernel`, measured the
   way Plan 11's t01 measured (32 calls per captured graph, noop floor
   reported), at 6–48 rows and at prefill sizes.
3. **Dispatch.** One knob, default off, routing the post-allreduce `mhc_post`
   through the fused kernel where the bench says it wins. Log which path each
   call site took (AGENTS.md §14: the knob being set is not the feature running).

The prize is bounded by `mhc_post_kernel`'s own 0.43 ms/step plus one launch
per call; the realistic saving is smaller.

### 4.3 QuickReduce for prefill-size messages (TODO #5)

No code change: `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` exists and defaults off,
with `VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB` and
`VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB` to bound it (names as recorded in
Plan 11's `T05_CANDIDATES_BCD.md`; confirm them on the base). Plan 11 measured
INT8 winning from 2,048 rows, 1.41× at 16,384 rows, at rel_l2 0.0082.

- QuickReduce is checked **before** AITER in the dispatch order, and it loses at
  every decode size (a fixed 23–27 µs codec floor). Set the minimum size so no
  decode-size message is admitted, and prove it from the trace.
- Arm INT8 only, the setting Plan 11 sized. The lower-precision modes carry
  larger error (INT4 measured rel_l2 0.122), and none was sized for prefill.
- This changes numerics, so it runs the full accuracy gate (§5) before any
  performance screen counts.

### 4.4 Not in scope

- **TODO #1**, two-stage below 16 rows: C8 only, priced between 0.075 and
  0.276 ms/step with the estimate contested, and its ≤ 8-row half sits behind
  vLLM's own allreduce, which nobody has benched. Revisit after 4.0 prices the
  C8 remainder.
- **TODO #2**, Engram's per-step all-gather: unpriced. Measure it first,
  separately.
- **Overlap:** needs a data-dependency analysis Plan 11 did not run.

## 5. Gates

Trace, on post-change captures of all four ranks, against 4.0's baseline:

- **4.1:** the >64 MiB allreduces run on `aiter::cross_device_reduce_2stage`, no
  RCCL allreduce above 64 MiB remains, and their summed time per step falls.
- **4.2:** the fused kernel replaces `mhc_post_kernel` at every post-allreduce
  position the knob covers, and the allreduce + post-mix time per step falls at
  both C8 and C32.
- **4.3:** QuickReduce appears only on prefill-size messages, never on decode.

Accuracy:

- **4.1:** per-element comparison against RCCL; a difference beyond bf16
  rounding stops the item.
- **4.2:** element-by-element match against the unfused pair. If the fused
  kernel is not bit-identical, it also needs full-length GSM8K with real DSpark
  block rejection, both arms.
- **4.3:** per-element tolerance against RCCL, then full-length GSM8K with real
  block rejection for both arms, as Plan 11 §4.4 specified.

Performance: Plan 09 §5, at C8 and C32, paired on one node with control screens
for the noise floor. Items expected below that floor (4.1) ship on the trace
and accuracy gates alone, as default corrections, and say so in the log.

## 6. Stop conditions

- **4.0:** the C8 remainder is arrival spread driven by host gaps. Hand it to
  Plan 14 with the per-rank evidence. This does not stop 4.1–4.3.
- **4.1:** the server ignores the setting Plan 11's bench used and the ceiling
  can only move with a code change to a constant in the image's AITER. Then
  carry it as a pinned patch, or drop the item if it cannot be pinned.
- **4.2:** the fused kernel computes something other than `MHCPostOp`, or hangs
  or corrupts under graph replay. Stop immediately in the second case; IPC
  buffer registration during capture is involved (Plan 14 §4.4).
- **4.3:** the accuracy gate fails, or QuickReduce admits any decode-size
  message that cannot be excluded by its size bounds.

## 7. Artifacts

```text
crusoe:/home/jiaweche/dsv41-profile-20260924/15_collectives_followup/
```

Persist the four-rank captures and the arrival-spread table, the fused-kernel
semantics check and bench, each knob diff with its hash, the accuracy
comparisons and GSM8K results, per-replay bench outputs, and the decision for
each item.

## 8. Execution setup

Written by the operator before submission. These are facts about the
environment, not choices about the plan.

### 8.1 Mesh, node and trees

- **Mesh.** This plan has its own mesh:
  - Session: `/home/jiaweche/Arbor-plan-executor-p15/sessions/plan-executor`.
  - Keeper: user unit `arbor-p15-keeper` on `slog-006`, capped at 1600M.
  - Agents: driver, `worker_1`, `worker_2` and the reviewer. There are two
    workers rather than three because `slog-006` also hosts Plan 12's mesh
    under the same per-user memory cap.
- **Node.** Job `174972` on `crsuse2-m2m-193` (`amd-burst`, 24 h, preemptible).
  It is this mesh's only node. The pool adopts jobs named `arbor-p15-*` and
  will not request another while it holds this one. The account allows four
  jobs per user, and all four are in use.
- **Another tenant's container** (`yzhou_model`) sits on this node. It held no
  GPU at allocation. Record GPU occupancy before every measured stage, as Plan
  09's `measure_one.sh` does.
- **Container.** `dsv41flash_arbor` runs the image by digest (`sha256:960228cf…`)
  with stock AITER 0.1.21.post2 and the fixed base `be794db46` overlaid.
  - Weights are node-local at
    `/mnt/m2m_nobackup/jiaweche/inferencex-dsv41flash/models/DeepSeek-V4.1-Flash`.
  - The aiperf client is at `/runtime/aiperf-src`.
  - The setup log is `$R/logs/setup.log`.
- **Not ours.** Never touch these jobs, their containers, or their sessions.
  Read-only access to their files is fine.

  | Job | Node | Belongs to | Mesh runs on | Session |
  | --- | --- | --- | --- | --- |
  | `174800` | `m2m-032` | Plan 10 | `slog-005` | `/home/jiaweche/Arbor-plan-executor` |
  | `174801` | `m2m-191` | Plan 12 | `slog-006`, the same login node as this mesh | `/home/jiaweche/Arbor-plan-executor-p12` |
  | `174831` | `m2m-016` | Plans 11 and 13 | `slog-007` | `/home/jiaweche/Arbor-plan-executor-p11` |

- **Paths.**
  - Artifacts (§7): `R=/home/jiaweche/dsv41-profile-20260924/15_collectives_followup`.
  - Base checkout: `/home/jiaweche/dsv41-merge/vllm-v6best-pr19`, a worktree of
    `/home/jiaweche/dsv41-merge/vllm`. Do not commit on it.
  - This plan's branch is a new worktree from `be794db46` (§3), for example at
    `/home/jiaweche/dsv41-merge/vllm-p15`.
- **Deploying Python changes.** Run
  `SRC=<worktree> WANT_SHA=<full sha> bash $R/bin/deploy_tree.sh` on the node.
  It overlays `vllm/`, hash-checks every `.py` in the commit against the
  container, confirms the compiled extensions are untouched, and imports the
  DSA attention modules. Anything short of `DEPLOY_OK sha=<sha>` means the tree
  is not what runs.
- **Changing AITER.** Plan 11 patched, rebuilt, installed and proved AITER's
  custom-allreduce module, with every step hashed. Its scripts are in
  `/home/jiaweche/dsv41-profile-20260924/11_tp_allreduce/run1/bin/`:
  `patch_aiter_car.py`, `build_aiter_car.sh`, `install_car.sh` and
  `prove_car_loaded.py`. Its review found two gaps:
  - the build script has no signal-safe restore, so an interrupted build can
    leave the package changed;
  - the in-place copy that installs a module into a live package was
    identified as a hazard but not fixed.
  
  If this plan changes AITER, recreate the container from the image afterwards
  rather than relying on a restore.
- **MoE configs.** The base's PR #19 MoE-config install writes its CSV into the
  AITER package inside the container, so every server after the first runs the
  FlyDSL MoE configs. That is identical in both arms of every A/B here and must
  stay so. Plan 09 owns it.
- **GPUs.** A server runs TP4 on GPUs 0–3. GPUs 4–7 are idle.
- **Other plans.** Plans 09, 11 and 14 (cited above) are in
  `/home/jiaweche/dsv41-profile-20260924/plans/`.

### 8.2 Existing work to reuse

- **Plan 11's evidence** is in
  `/home/jiaweche/dsv41-profile-20260924/11_tp_allreduce/`:
  - `T09_SETTLEMENT.md`, `T03_CALL_SITES.md` and `T05_CANDIDATES_BCD.md`;
  - raw bench outputs under `run1/`;
  - its graph-captured allreduce bench `run1/bin/car_bench.py`, which times K
    calls per replay against a no-op floor;
  - the QuickReduce runs `run1/bin/t02_*`.
- **Plan 11's mesh records.** Its TODO (the numbered entries this plan cites)
  is `/home/jiaweche/Arbor-plan-executor-p11/sessions/plan-executor/mesh/TODO.md`,
  with `todo.jsonl` beside it. Its review is `mesh/reviews/p01-reviewer.md`,
  with a pricing addendum beside it. That mesh is live and running Plan 13:
  read, never write.
- **Captures of this base.**
  - Plan 10's knob-off captures at C32 and C8, 400 `ProfilerStep#` each:
    `/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/cap_base_c{32,8}/traces/`.
  - Only rank 0's trace reached `/home`, from these and from Plan 09's
    captures. The profiler does write all four ranks: Plan 09's
    `capture_one.sh` waits for four trace files in the container's
    `/tmp/traces`. §4.0 needs all four copied out.
  - Traces are 30–200 MB compressed per rank. `/home` is over 90% full: keep
    raw traces on node-local disk and copy summaries, plus whatever the
    decision needs.
- **Scripts.** Copy and change what differs:
  - Plan 09 (`.../09_moe_experts/run1/bin/`): `capture_one.sh`,
    `measure_one.sh` (one A/B point: fresh control server, marker check,
    discarded warmup, one measured replay), `eval_one.sh` (GSM8K with real
    block rejection), `p09_driver.sh` and `summarize_ab.py`.
  - Plan 10 (`.../10_dsa_indexer_decode/run1/bin/`): `dsa_attrib.py` (Kineto
    attribution, including graph-replayed kernels) and `t01_driver.sh`
    (captures under a user unit).
- **Harness.** Read, never edit; it belongs to another mesh's session:
  `/home/jiaweche/Arbor-dsv41-mesh/sessions/megamoe-v9/mesh/toolbox/crusoe/{boot_control_server.sh,replay_arm.sh}`.

### 8.3 Things that have already cost time here

- **What Plan 11's review found.** The reviewer upheld Plan 11's stop but
  rejected several of its conclusions. Avoid repeating these:
  - **Equal error is not equal output.** Equal maximum error against a
    reference does not show two arms produce equal tensors. Compare the arms
    directly, element by element, against a tolerance declared before the run.
  - **A gate that was not run is "not run".** It is not "passed", even after a
    stop makes it moot.
  - **Keep the raw samples.** Keep each microbenchmark replay's samples, and
    record the exact script that produced them.
  - **Step counts come from `ProfilerStep#`.** Every capture here records 400.
    Never infer them from median latency.
  - **Timing overlap is not independence.** Two kernels not overlapping today
    says nothing about whether one depends on the other.
  - **Handle every shape.** Plan 11's attribution parser dropped
    three-dimensional (Engram) shapes, which produced wrong byte counts.
    Handle every shape, or report the ones you could not.
- **Memory.** Parse traces on the node through
  `srun --overlap --jobid=174972`, never on the login node.
  - An agent that parsed a Kineto trace on a login node grew to 1.8 GB, and
    every process of this user on that host slowed to a crawl.
  - This mesh's unit is capped at 1600M, and a process that exceeds it is
    OOM-killed.
  - The login node caps the user at 4 GiB in total, shared with Plan 12's mesh.
- **The login node kills your background processes.**
  `/usr/local/sbin/shared-host-watch` SIGKILLs everything left in an ssh
  session's scope about a minute after the session ends. Anything that must
  outlive a command goes in a user unit:
  `systemd-run --user --collect --unit=<name> bash -lc '<cmd>'`. Processes the
  agents start inherit the keeper's unit.
- **Login nodes.** `slog-005` refuses new logins while its memory pool is
  full. `slog-003` and `slog-004` accept only interactive sessions, with a
  512 MiB per-user cap.
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
- **`pgrep -f` matches itself.** Inside `bash -lc "..."`, `pgrep -f` matches its
  own command line. Use the `[m]easure_one` form.
- **Heredocs.** A heredoc into `docker exec` needs `-i`.
- **Liveness.** A driver that records `STATUS=RUNNING` and then dies looks
  healthy. Check the pid or unit, not the status file.
- **Seed.** The InferenceX mount hardcodes seed 42 (Plan 09 §5).
