# DeepSeek V4.1 Flash BF16 GEMM Tuning Plan

> One of six profile-driven plans (09–14) from the MI355X C32 deep trace of
> 2026-09-24. The source profile, fixed baseline and performance rules are
> shared and kept once, in [Plan 09](./09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md)
> §0, §3 and §5.
> This is an internal execution document, not published InferenceX documentation.

Status: **submitted** to the Plan 11 executor mesh on 2026-09-25 (setup in §8)

> **Base changed (2026-09-24)** to `arbor-v2-plus-v6best-pr19` (Plan 09 §3).
> The dispatch path below was read at `93205f244`. Recheck it on the new base
> before execution.

## 1. Objective

Stop the two prefill BF16 GEMM shape families from falling to untuned Tensile
fallbacks, and first find out whether doing so is worth anything.

## 2. Evidence

The server log emits this line **47,292** times over the 3h26m run, across
four ranks:

```text
not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv,
will use default config! using torch solution:0
```

Every miss falls into one of exactly two shape families:

```text
M:<prefill token count>, N:32,  K:5120
M:<prefill token count>, N:128, K:512
```

The fallbacks run on Tensile-generated `Cijk_*` kernels:

| kernel | calls | ms |
| --- | ---: | ---: |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_*` | 17,759 | 212.31 |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x128_MI16x16x1_*` | 2,010 | 37.88 |
| whole `Cijk_*` family | 24,354 | 404.38 (**3.64%**) |

These are not the fp32 router GEMM that
[PR #19](https://github.com/qianghan-amd/vllm-eed1f3d0/pull/19) item 4 fixes.
That is a different table and a different shape.

### 2.1 Why the table never hits

`rocm_unquantized_gemm_impl` (`vllm/model_executor/layers/utils.py:344`) picks
the first path that applies:

1. the skinny kernels, gated on small token counts;
2. AITER's Triton `gemm_a16w16`, for whitelisted shapes;
3. AITER `tgemm.mm`, when `VLLM_ROCM_USE_AITER_LINEAR` is set (the default).

`tgemm` looks up `bf16_tuned_gemm.csv` by exact M. M here is the prefill token
count, so every distinct prefill size misses, and the table cannot hit by
construction.

### 2.2 The size of the prize is not established

The trace-analysis note calls the 16×16 macro tile "badly under-tiled". That
has not been checked, and there is a reason to doubt it:

- **The kernel may already be near its memory floor.** At N = 32 the GEMM's
  traffic is almost entirely the M × 5120 bf16 activation, 10 KiB per row.
  `MT16x16x1024` averages **12.0 µs** per call (212.31 ms / 17,759). Reading
  5,000 rows is 51 MB, about 10 µs at 5 TB/s. At typical M this kernel may
  already be close to the floor.
- **The upper bound is small.** It is 3.64% of GPU time, and the realistic gain
  is some fraction of that. Size it (step 4.2) before tuning anything.

## 3. Fixed baseline

As Plan 09 §3. Only BF16 GEMM dispatch for these two shape families may change.

## 4. Work items

### 4.1 Name the layers

Find the BF16 projections with `(N=32, K=5120)` and `(N=128, K=512)` in
`vllm/models/deepseek_v4_1/`, and confirm that they reach
`rocm_unquantized_gemm`. The trace-analysis note identifies them as DSA indexer
GEMMs, but it did not trace them to source. Take the M distribution from
recorded shapes in the deep trace.

### 4.2 Roofline per call

For each call, compute the activation-read floor at its M and compare it with
the measured time. Sum the recoverable time. If that sum is under the noise
floor at C32, stop here (§6).

### 4.3 Candidates, one knob each, default off

- **A. M-bucketed lookup.** Tune a bucket set with AITER's GEMM tuner and round
  M up to the next bucket at lookup. Measure the padding cost.
- **B. Route to Triton.** Whitelist the two families to `gemm_a16w16`, or to a
  skinny path, if either beats the Tensile fallback at the observed M.
- **C. Fuse the N = 32 projection** into another GEMM that reads the same
  hidden-state activation, by appending columns. That removes one full read of
  the activation. A dtype mismatch (BF16 against MXFP8) makes this an accuracy
  question, so it needs the gate in §5.

### 4.4 Host side of the miss

Determine whether AITER performs the lookup and logs per call or once per new
shape. 47,292 lines is about 11,800 per rank. If the miss path runs on every
call, a miss cache is a host-side win independent of the GPU work.

## 5. Gates

- **Numerics.** A and B must match the BF16 reference within GEMM tolerance.
  C changes precision, so it runs the 1,319-example GSM8K with real DSpark block
  rejection, and must not change the indexer's selected positions beyond ties
  (see Plan 10 §4.2 for the index-set comparison).
- **Trace.** `Cijk_*` time for these families per profiler step falls below the
  baseline 1.01 ms/step (404.38 ms / 400), and the miss line disappears from the
  server log.
- **Performance.** Plan 09 §5. The effect lives in prefill, so expect it at C32
  first and check that C8 does not regress.

## 6. Stop conditions

- 4.2 shows the recoverable time is under the C32 noise floor. The finding is
  then that the kernel was not under-tiled in any sense that matters, and
  only 4.4 remains.
- C moves the indexer's selected positions.

## 7. Artifacts

```text
crusoe:/home/jiaweche/dsv41-profile-20260924/13_bf16_gemm_tuning/
```

Persist the layer map, the M distribution, the roofline table, any tuned
bucket CSV with its hash, the knob diffs, trace captures, per-replay bench
outputs and the decision.

## 8. Execution setup

Written by the operator before submission. These are facts about the
environment, not choices about the plan.

### 8.1 Node and trees

- **Mesh and node.** This plan runs on the mesh that executed Plan 11: session
  `/home/jiaweche/Arbor-plan-executor-p11/sessions/plan-executor`, keeper
  `arbor-p11-keeper` on `slog-007`. Its node is job `174710` on
  `crsuse2-m2m-250` (`amd-burst`, 24 h, preemptible). The pool adopts jobs
  named `arbor-p11-*` and will not request another while it holds this one.
- **A fresh container.** Plan 11 rebuilt AITER's custom-allreduce module
  inside the old container. `dsv41flash_arbor` was therefore recreated from
  the image by digest (`sha256:960228cf…`) at 13:59 UTC on 2026-09-25, and the
  fixed base `be794db46` was overlaid again. AITER in it is stock 0.1.21.post2,
  and its JIT caches start cold.
  - Weights are node-local at
    `/mnt/m2m_nobackup/jiaweche/inferencex-dsv41flash/models/DeepSeek-V4.1-Flash`.
  - The aiperf client is at `/runtime/aiperf-src`.
  - The setup log is `$R/logs/setup.log`.
- **Not ours.**
  - Job `174441` on `m2m-016` is Plan 09's A/B.
  - Job `174682` on `m2m-031` is Plan 10's node, and Plan 10's mesh runs on
    `slog-005` from `/home/jiaweche/Arbor-plan-executor`.
  - Never touch these jobs, their containers, or that session. Read-only
    access to their files is fine.
- **Paths.**
  - Artifacts (§7): `R=/home/jiaweche/dsv41-profile-20260924/13_bf16_gemm_tuning`.
  - Base checkout: `/home/jiaweche/dsv41-merge/vllm-v6best-pr19`, a worktree of
    `/home/jiaweche/dsv41-merge/vllm`. Do not commit on it.
  - This plan's branch: make a new worktree from `be794db46`, for example
    branch `arbor-v2-plus-v6best-pr19-p13` at
    `/home/jiaweche/dsv41-merge/vllm-p13`. `vllm-p10*` and `vllm-p11` belong
    to other plans.
- **Deploying Python changes.** Run
  `SRC=<worktree> WANT_SHA=<full sha> bash $R/bin/deploy_tree.sh` on the node.
  It overlays `vllm/`, hash-checks every `.py` in the commit against the
  container, confirms the compiled extensions are untouched, and imports the
  DSA attention modules. Anything short of `DEPLOY_OK sha=<sha>` means the tree
  is not what runs.
- **Injecting a tuned CSV into AITER without writing into its package.** Plan
  09's knob (`cbfdad177`, `_install_dsv41flash_flydsl_moe_configs` in
  `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`) is a working example:
  - it writes a private CSV;
  - appends it to AITER's config list environment variable;
  - clears AITER's config caches;
  - asserts that the rows reached the merged table.
  
  Writing into `aiter/configs/` instead leaks into every later process in the
  container, control arms included.
- **MoE configs.** The base's PR #19 MoE-config install writes its CSV into the
  AITER package inside the container, so every server after the first runs the
  FlyDSL MoE configs. That is identical in both arms of every A/B here and must
  stay so. Plan 09 owns it.
- **GPUs.** A server runs TP4 on GPUs 0–3. GPUs 4–7 are idle.
- **Other plans.** Plan 09 (§0, §3, §5 above) and Plan 10 (§4.2's index-set
  comparison, cited in §5) are in `/home/jiaweche/dsv41-profile-20260924/plans/`.

### 8.2 Existing work to reuse

- **Traces of this base, readable now.**
  - Plan 10's fresh knob-off captures of `be794db46` at C32 and C8, 400
    `ProfilerStep#` each:
    `/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/cap_base_c{32,8}/traces/`.
  - Plan 09's captures:
    `/home/jiaweche/dsv41-profile-20260924/09_moe_experts/run1/trace_a{0,1}_c{8,32}/traces/`.
  - Only the rank-0 trace of each is on `/home`.
  - Shapes are recorded for eager ops only; graph replays record none. Plan 09
    §8 buckets eager MoE calls by M this way. Anything compared against a
    candidate is captured on this node.
- **Plan 10's working index-set comparison** (§5's numerics gate for C) is
  `/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/bin/topk_bench.py`,
  with its report under `run1/`.
- **Scripts.** Copy and change what differs:
  - Plan 09 (`.../09_moe_experts/run1/bin/`): `capture_one.sh`,
    `measure_one.sh` (one A/B point: fresh control server, marker check,
    discarded warmup, one measured replay), `eval_one.sh` (GSM8K with real
    block rejection), `p09_driver.sh` and `summarize_ab.py`.
  - Plan 10 (`.../10_dsa_indexer_decode/run1/bin/`): `dsa_attrib.py` (Kineto
    attribution, including graph-replayed kernels), `t01_driver.sh` (captures
    under a user unit), and knob probes.
  - Plan 11 (`.../11_tp_allreduce/run1/bin/`): graph-captured microbenchmarks
    that time K calls per replay against a no-op floor.
- **Settled decisions and reviews.**
  - Plan 10's decisions:
    `/home/jiaweche/Arbor-plan-executor/sessions/plan-executor/mesh/intake/10_DSV41_FLASH_PROFILE_DSA_INDEXER_DECODE.decisions.jsonl`.
  - Plan 11's decisions, settlement and review are in this mesh's session. The
    review is `mesh/reviews/p01-reviewer.md`.
- **Harness.** Read, never edit; it belongs to another mesh's session:
  `/home/jiaweche/Arbor-dsv41-mesh/sessions/megamoe-v9/mesh/toolbox/crusoe/{boot_control_server.sh,replay_arm.sh}`.
- **The Ruby deep trace** behind §2 (and §4.1's "deep trace") is on `ruby-1`,
  which Crusoe cannot reach, so §2's numbers are the reference, not a file.

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
- **The login node kills your background processes.**
  `/usr/local/sbin/shared-host-watch` SIGKILLs everything left in an ssh
  session's scope about a minute after the session ends, and ends every session
  at 8 hours. Anything that must outlive a command goes in a user unit:
  `systemd-run --user --collect --unit=<name> bash -lc '<cmd>'`. Lingering is
  enabled on `slog-007`. Processes the agents start inherit the keeper's unit.
- **Memory on the login node.** It caps the whole user at 4 GiB per login node,
  with throttling from 2.5 GiB. Parse traces and run tuners on the node through
  `srun --overlap --jobid=174710`, never on the login node.
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
