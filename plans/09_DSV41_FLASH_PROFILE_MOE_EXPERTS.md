# DeepSeek V4.1 Flash MoE Expert Kernel Plan

> One of six profile-driven plans (09–14), each attacking one cost centre of the
> MI355X C32 deep trace of 2026-09-24. They share the baseline and measurement
> rules below and are otherwise independent. They are not children of the
> [MegaMoE master plan](./01_DSV41_FLASH_MEGAMOE_AGENTX.md): that track uses a
> different vLLM base and topology.
> This is an internal execution document, not published InferenceX documentation.

Status: **full set rejected, reduced set neutral** — all 16 tiers: C32 0.9650,
C8 1.0026. Without the `_xcd4` tiers (32–2048): C32 1.0016, C8 1.0028. GSM8K
neutral. No end-to-end win (see §8)

## 0. Source profile

| | |
| --- | --- |
| trace | Ruby job 44278, `cv350-rck-g03-e14-08`, C32, TP4, seed 42 |
| tree traced | `merge/v2v6-knobs` @ `5f99d56f0` + `VLLM_DSV41_FUSE_ACT_QUANT=1` + `VLLM_ROCM_DSA_VARCTX_SCHEDULE=1` |
| window | step-capped, about 45 minutes into an 11,000 s replay; 11,103.6 ms GPU kernel time over 400 `ProfilerStep#` annotations, 27.76 ms/step |
| run | 332.34 tok/s/GPU, 13,194 / 13,552 requests, KV hit 0.968, profiler attached |
| files | `ruby-1:/home/jiaweche/mi355x-traces/c32_4h/` (`json/rank{0..3}_trace.json`, `server.log`) |
| analysis | `jiaweichen-amd/agent/HANDOFF-dsv41-trace-analysis.md`; scripts `agent/kernel_hist.py`, `agent/kernel_percentiles.py`, `agent/launch_hist.py` |

**Per-step figures.** The trace-analysis handoff divides by 861 steps. The trace
holds 400 `ProfilerStep#` annotations, matching the capture's
`active_iterations=400`, so the handoff's per-step values are 2.15× too small.
Ratios between them are unaffected. Every per-step number in plans 09–14 uses 400.

**Tree.** The benchmark base is now `arbor-v2-plus-v6best-pr19` (§3).
`5f99d56f0` is `arbor-v2-plus-v6best` plus 34 commits touching 18 files under
`vllm/`, so the traced tree is not the base. The two knobs on in the trace
(act-quant fusion and the DSA VARCTX schedule) do not touch the routed-MoE path,
so the MoE evidence here carries over. The other plans must recheck their own
components.

Two further limits apply to every share below:

- **C32 only.** No MI355X C8 trace exists. On B200 the C8 decode window runs a
  different graph variant and GEMM mix from C32 (`agent/arbor_dashboard/b200-c8-vs-c32.md`),
  so a C32 share predicts nothing at C8.
- **Three v6lean defaults were off when traced:** `PREFILL_KEY_TILED`,
  `GFX950_FLYDSL_LOGITS`, `ENGRAM_HOST_OFFLOAD`. At C8 the first two decline on
  shape and Engram is never constructed; at C32 this has not been checked.

## 1. Objective

Reduce the time spent in routed-expert MoE kernels, the largest block in the
trace, at both C8 and C32, without changing model accuracy.

## 2. Evidence

MoE is **20.56%** of GPU kernel time, all on the AITER asm path
(`aiter::fused_moe_`, 520 eager invocations in the window), not FlyDSL:

| kernel | calls | ms | µs/call | share |
| --- | ---: | ---: | ---: | ---: |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t32x128x256_pm1_async_gui_v33` | 16,724 | 1111.21 | 66.4 | 10.01% |
| `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x128x128_vscale_fix3_fp4opt_v1_pm1` | 16,724 | 733.46 | 43.9 | 6.61% |
| other `mfma_moe1` tiles (t128, t64) | 520 | 275.65 | ~530 | 2.48% |
| other `mfma_moe2` tiles (t128, t64) | 520 | 162.59 | ~313 | 1.46% |
| **total** | **34,488** | **2282.91** | | **20.56%** |

The t128/t64 call count equals the eager invocation count, so those are
probably prefill and mixed batches; the 16,724-call t32 kernels are graph
replays, probably decode. On that reading decode MoE is about 16.6% and
prefill about 3.9%. Step 4.1 must confirm it before any work is sized.

[PR #19](https://github.com/qianghan-amd/vllm-eed1f3d0/pull/19) (open, head
`901b45efa`) item 2 injects a FlyDSL tuned CSV (WPE=4, FP8 stage 1) into
`aiter.fused_moe.cfg_2stages` at import, which should replace the `mfma_moe*`
kernels with `flydsl_moe*`. PR #19 is merged into the new base (§3), and §4.3's
knob makes this item switchable, so it can be measured against its own absence.

The PR reports 287.4 → 311.3 tok/s/chip (+8.3%) for items 2, 3 and 4 together.
That number cannot be carried over:

- its baseline already contains item 1;
- it is one seed, one replay, no stated warmup, `DURATION=600` with 75.4% of
  requests completing;
- item 4 (the `(5120, 384)` router GEMM entry) conflicted with v6best's own
  entry for that shape (§3). It is in both arms of this plan's A/B, so none of
  this plan's measurement is attributable to it.

## 3. Fixed baseline

```text
base:      arbor-v2-plus-v6best-pr19 @ be794db46
           = arbor-v2-plus-v6best @ 8f400a0ac (V2+V6) + merge of PR #19 (901b45efa)
           crusoe:/home/jiaweche/dsv41-merge/vllm-v6best-pr19 (worktree; not pushed)
plan 09:   arbor-v2-plus-v6best-pr19-p09 @ cbfdad1774a5a0ceefc667dc5f01a5dbc577670e
           = base + VLLM_DSV41_FLYDSL_MOE_CONFIGS knob (§4.3)
           crusoe:/home/jiaweche/dsv41-merge/vllm-p09 (worktree; not pushed)
image:     vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657
server:    TP4 / EP1 on GPUs 0-3, --moe-backend aiter, max-num-seqs 128,
           max-num-batched-tokens 16384, max-model-len 1048576,
           max-cudagraph-capture-size 1024, gpu-memory-utilization 0.9
spec:      DSpark, num_speculative_tokens 5, synthetic acceptance 3.51
harness:   Crusoe p04 control harness (boot_control_server.sh, replay_arm.sh),
           server_env_blind.list md5 64a08abe4e522e6ba451e4b6bebdebcb
workload:  AgentX semianalysis_cc_traces_weka_062126, NUM_PROMPTS 320,
           DURATION 600, C8 and C32
```

Only the MoE expert kernel selection may change.

The merge conflicted in `rocm_fp32_router_gemm.py`, where both sides added
`(5120, 384)`. The resolution keeps v6best's per-shape cap table and takes PR
#19's launch configs and its 32-token bound. v6best had bounded that shape at 16
tokens with a different launch table, and the two sides' measurements disagree.

## 4. Work items

### 4.1 Attribute before changing anything

- From the rank-0 deep trace (`record_shapes` was on), split MoE time into
  graph-replayed decode and eager prefill, and record tokens per call and
  experts touched per call.
- Compute achieved weight bandwidth per decode call (touched expert bytes over
  kernel time) against MI355X HBM peak. Bandwidth-bound decode MoE only gets
  faster by moving fewer bytes; occupancy-bound decode MoE responds to tiling
  and waves per EU, which is what the FlyDSL configs change.
- Capture a C8 deep trace on `93205f244` with the same step-capped recipe
  (`ruby-1:/home/jiaweche/mi355x-traces/batch_4h.sbatch`, `CONC=8`). One C8
  capture serves all of plans 09–14. The profiler collects once per server
  lifetime, so never probe `/start_profile` first, and never book throughput
  from a profiled server.

### 4.2 Kernel bench per token tier

Using the expert-token distributions from 4.1, bench the current asm kernels
against the PR #19 FlyDSL configs standalone, at the C8 decode shape
(48 verify rows × 6 experts), the C32 decode shape (192 × 6), and each prefill
tier observed. Record the winner per tier.

### 4.3 Port behind a knob, fixing what the PR leaves unsafe

Add `VLLM_DSV41_FLYDSL_MOE_CONFIGS`. It defaults to `1` on the plan branch,
because the base already contains PR #19; the control arm sets it to `0`. Beyond
the PR's logic:

- **Fail loudly.** The PR wraps the install in `except Exception: pass`, so a
  failed install silently measures the old path. Log the injected tier count on
  success, the reason on failure, and a distinct marker when the knob is off.
- **No import-time side effects.** The PR writes into the installed aiter
  package's `configs/model_configs/` and deletes
  `/tmp/aiter_configs/tuned_fmoe.csv` at import. Install from a private
  directory at engine init instead, and log its path and hash.
- **Inject only winning tiers.** If 4.2 shows the asm kernel winning at a tier
  (the C8 decode tier is the likely one), leave that tier alone.

### 4.4 Accuracy

FP8 stage 1 changes activation precision in the expert GEMM. Run the
1,319-example GSM8K with real DSpark block rejection, as Plans 07–08 did, for
both arms. The AgentX benchmark uses synthetic acceptance and cannot detect an
acceptance-rate change.

## 5. Gates

Kernel: each injected tier is faster than asm at its own shape.

Trace, on a post-change capture:

- `flydsl_moe1_*` / `flydsl_moe2_*` replace `mfma_moe1_*` / `mfma_moe2_*` at
  every injected tier and nowhere else;
- MoE time per profiler step falls below the knob-off capture's own figure,
  taken on the same node. For reference, the deep trace's expert-GEMM kernels
  were 5.71 ms/step (2282.91 ms / 400). Compare per step, because windows differ
  in length.

Performance, the same for plans 09–14:

- Knob markers: both states are logged and matched by the harness capture
  pattern, asserted offline before the first screen (AGENTS.md §14).
- Pairing: control is the plan's branch with its knob off and the candidate is
  the same tree with the knob on, using the same image on the same node. Each
  replay gets a
  fresh server, a discarded warmup and one measured replay, alternating arms.
  Never compare across nodes: the same C8 baseline read 75.8–78.2 on three
  nodes.
- Both points: report C8 and C32. A C32 gain paid for with C8 is a Pareto
  point, not a win.
- Noise floor: compute the threshold from same-node control screens before the
  first candidate. For reference, v6lean A/B arm spreads were 0.65–0.73% at C8
  and 0.63–1.19% at C32.
- Seed: the Crusoe InferenceX mount (`InferenceX-dsv41-mesh` @ `16fee0ab`)
  hardcodes `--random-seed 42` and ignores `BENCH_SEED`. Seed-42 screens may
  reject a change. A claim needs three seeds, which needs a mount whose
  `build_replay_cmd` reads `${BENCH_SEED:-42}`, for both arms.
- Validity: duration band only. Record `duration_seconds`,
  `num_requests_successful` and `kv_cache_hit_rate` per replay, each replay in
  its own output directory.

## 6. Stop conditions

- The FlyDSL install cannot be made to log its own success and failure.
- 4.2 shows no tier where FlyDSL wins.
- The kernel wins but end-to-end loses at either point. Profile that pair
  before anything else; launch-width changes have inverted at low concurrency on
  this model before (AGENTS.md §8).
- GSM8K regresses.

## 7. Artifacts

```text
crusoe:/home/jiaweche/dsv41-profile-20260924/09_moe_experts/
```

Persist the attribution tables, the kernel bench, the installed CSV with its
hash, the knob diff, both trace captures, per-replay bench outputs, and the
decision.

## 8. Execution log

### 2026-09-24

**Branches.** Base `be794db46`, plan branch `cbfdad177` (§3). Neither is pushed.

**4.1 attribution.** From the rank-0 deep trace, using
`ruby-1:/home/jiaweche/mi355x-traces/analysis/moe_attrib.py`. MoE-named
kernels total 2,733.96 ms, counting expert GEMMs plus sort, quant and gating.
83.0% of that is graph-replayed and 17.0% is eager. Split by graph class
(`moe1` calls per `hipGraphLaunch`):

| graph class | launches | `moe1` p50 µs | `moe2` p50 µs | expert-GEMM ms |
| --- | ---: | ---: | ---: | ---: |
| 40 (target verify graph) | 379 | 64.1 | 44.3 | 1,696.99 |
| 3 | 400 | 18.8 | 12.1 | 42.30 |
| 1 | 320 | 200.2 | 113.5 | 99.78 |

The 40-layer verify graph alone is 74% of expert-GEMM time. Eager expert GEMMs,
by token count M:

| M bucket | calls | tiles | ms |
| --- | ---: | --- | ---: |
| ≤2048 | 160 | t64 | 92.01 |
| ≤4096 | 280 | t128 | 227.37 |
| ≤8192 | 40 | t128 | 44.12 |
| ≤16384 | 40 | t64 | 95.73 |

Experts touched per call and achieved bandwidth are not measurable from this
trace. It records no routing, and graph replays record no shapes.

**How AITER keys the tiers.** In the image (AITER 0.1.21.post2),
`get_padded_M` rounds M below 32,768 up to the next power of two before the
config lookup:

- C32 verify (192 rows) uses the 256 tier;
- C8 verify (48 rows) uses the 64 tier;
- PR #19's 192 row can never match.

**What PR #19's install does in this image.**

- It writes its CSV into `aiter/configs/model_configs/`, and AITER's default
  config list picks that directory up in every later process. After one run,
  every later server in the container gets FlyDSL whatever the code says,
  including a control arm.
- At import it replaces `cfg_2stages`, which is still unloaded at that point,
  with only its own 16 rows. AITER then never loads its own `tuned_fmoe.csv` in
  that process.

**Knob (`cbfdad177`).**

- Writes a private CSV and appends it to `AITER_CONFIG_FMOE`, rebuilding AITER's
  default list first.
- Clears AITER's config caches.
- Asserts that all 16 rows reach the merged table, and raises otherwise.

Verified in the container:

- knob 0: 0 DSV4.1 rows in the merged table;
- knob 1: 16 rows;
- knob 0 after a knob-1 process: 0 rows;
- nothing written into the AITER package.

**Deviation from 4.2.** The standalone kernel bench is replaced by a trace
comparison at production shapes: one profiled capture per arm at C8 and C32, on
the same node, with real routing. A standalone bench would need weights and
scales rebuilt in vLLM's preshuffled layout, and would route random tokens
unlike production.

**Node.** Job 174183 on `crsuse2-m2m-037`. Two earlier nodes were unusable:

- `m2m-093`: another tenant's vLLM server held all eight GPUs.
- `m2m-249`: Docker's `containers/` directory was missing, so no container
  could be created.

`m2m-037` carries five other tenants' containers, none holding a GPU at
bring-up, so every measured point records GPU occupancy first. The node's login
profile also points `HF_HOME` at a read-only NFS mount, which bring-up unsets.

**Pipeline.** `run1/bin/p09_driver.sh` runs:

1. the trace gate: 4 captures, stopping on failure;
2. GSM8K for both arms, with real block rejection;
3. the A/B: two rounds × {C32, C8} × two arms, with arm order alternating
   between rounds.

It started at 23:48 UTC.

### 2026-09-25: trace gate

**Harness fixes before the gate could run.** Three problems, all in the run
scripts rather than in the tree:

1. A stray trailing carriage return turned a successful boot into exit 127.
2. A steady-state trigger waited for 24 of 32 requests to be running.
   AgentX occupancy sits at 1–6 for long stretches, so it never fired. The
   profile now starts 180–420 s after the first request.
3. The driver read each step's `STATUS` from the login node right after the
   worker node wrote it to NFS, before the login node's cache had caught up.

**Gate criterion corrected.** The first gate required FlyDSL-named kernels and
zero `mfma_moe` calls with the knob on, and failed. AITER's log shows the
injected configs were selected for every tier the model hit (`using 2stage
(kernelName1='flydsl_moe1_…')` for `(…, 5120, 640, 384, 6, …)`). But the kernels
they launch are still named `mfma_moe*`: the change shows up as the FP8-output
stage-1 variant (`mfma_moe1_silu_mul_afp8_wfp4_fp8_…_fp8q_sort…`) and the
`_xcd4` stage-2 variant. The DSpark draft model's MoE (128 experts, top-3) has
no injected config and keeps the bf16-output kernels in both arms. The gate now
checks for FP8-output stage-1 kernels with the knob on and none with it off. It
**passes** at C8 and C32.

**Kernel-level comparison** (one profiled window per arm, 400 steps, seed 42):

- **C8, matched windows** (390 against 391 verify-graph launches). MoE-named time
  falls 4.9% per step, 2.910 → 2.766 ms. The saving is the separate
  activation-quantize kernel that FP8-output stage 1 removes; the expert GEMMs
  themselves are 1.8% slower (2.018 → 2.055 ms/step).
- **C32, unmatched windows.** The knob-off window caught a prefill burst (111
  verify-graph launches, eager MoE 33.9 ms/step) and the knob-on window did not
  (381 launches). Only per-call comparisons hold:
  - eager prefill at M 2,049–16,384 is 9–31% faster per call;
  - eager prefill at M ≤ 2,048 is 18–29% slower (tiers 1024 and 2048 select t32
    tiles);
  - verify-graph stage 1 has p50 46.9 → 42.6 µs, at different occupancy;
  - the `_xcd4` stage-2 variant used at tiers 32–256 looks slower than the
    non-`xcd4` kernel it replaces.

**Decision.** Proceed to accuracy and the end-to-end A/B with PR #19's full
config set as merged, since that set is what the base branch carries. The
per-tier readings above are noisy (one window, occupancy not matched at C32),
so no tier is dropped on them. The follow-up candidate is the same configs
without tiers ≤ 2048 and without the `_xcd4` stage-2 kernels, measured the same
way.

### 2026-09-25: accuracy (§4.4)

Full 1,319-example GSM8K. Both servers ran real DSpark block rejection
(`rejection_sample_method: block`), and the knob markers confirmed each arm:

| arm | strict match | flexible extract |
| --- | ---: | ---: |
| knob 0 (AITER defaults) | 0.9742 ± 0.0044 | 0.9735 ± 0.0044 |
| knob 1 (PR #19 FlyDSL configs) | 0.9727 ± 0.0045 | 0.9719 ± 0.0045 |

The −0.15-point difference is inside one standard error of either arm, so
there is no accuracy regression. The paired A/B started at 02:26 UTC.

### 2026-09-25: node lost, A/B restarted

Every `amd-burst` allocation we held was cancelled between 03:42 and 03:43
UTC, with `Reason=None`, about four hours before the 24-hour limit. That
included job 174183 on `m2m-037`. Other users' jobs were on two of the nodes
within minutes.

One A/B point had completed before that: C32 with the knob off read 331.603
tok/s/GPU (1,115 requests, duration 622.6 s, KV hit 0.9088). The knob-on point
was cancelled during its measured replay. Its pair can only be measured on the
same node, so that point is set aside (`ab_r1_c32_a0.m2m-037`), not used.

The resume allocated a new node. `m2m-243` was rejected because another
tenant's processes held its GPUs; `m2m-016` (job 174441) was taken. On it the
resume provisions the node, redeploys `cbfdad177`, and restarts the driver.
The trace gate and accuracy results carry over, since they are not node-paired;
all eight A/B points run again on the new node.

### 2026-09-25: A/B round 1 on `m2m-016`

Seed 42. Each point used a fresh server, a discarded warmup and one measured
replay. GPU occupancy was recorded before each stage and showed only our own
server's four workers.

| conc | knob | tok/s/GPU | p50 TPOT ms | p90 TPOT ms | p90 TTFT s | KV hit | reqs | duration s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0 | 298.390 | 9.41 | 17.20 | 1.582 | 0.9052 | 1060 | 628.2 |
| 32 | 1 | 285.674 | 10.26 | 17.00 | 1.533 | 0.9036 | 1044 | 629.7 |
| 8 | 0 | 76.267 | 3.87 | 4.51 | 0.760 | 0.9186 | 226 | 601.1 |
| 8 | 1 | 76.466 | 3.89 | 4.60 | 0.752 | 0.9175 | 226 | 599.6 |

Knob on / off: **C32 0.9574**, **C8 1.0026**. The C32 loss is about four times
the widest v6lean arm spread (1.19%). It shows up as p50 TPOT, 9.41 → 10.26 ms,
i.e. decode, not prefill, and is consistent with the trace reading that the
`_xcd4` stage-2 kernel used at tiers 32–256 is slower. Round 1 is one pair per
point; the verdict waits for round 2.

`m2m-016` reads about 10% below other nodes at C32 (298.4 here against 331.6
on `m2m-037`). That affects nothing paired on this node, but no number from
it should be compared with another node's.

**Driver death.** The login-node driver on `slog-005` died silently after
starting round 2's first point (06:38 UTC). The node-side step kept running and
finished: C32 knob 1 read **289.868** (p50 TPOT 10.14 ms, 1,043 requests,
616.4 s), close to round 1's 285.674. `slog-005` had not rebooted, and its user
cgroup showed no OOM, but none of our processes survived. The cause is not
established. The point was copied from node-local storage and the driver was
restarted at 11:05 UTC on the same allocation, so round 2 stays paired with
round 1. From then on a guard off the cluster checked the driver every five
minutes and could restart it up to three times.

### 2026-09-25: A/B round 2 and verdict

**Harness.** The restarted driver died again within minutes. `slog-005` kills
every session-scoped process of a user shortly after that user's last ssh
session closes. This still happens with `loginctl enable-linger`. The
earlier driver lived for hours only because a session happened to stay open.
A process started with `systemd-run --user` survives, because it belongs to
the user's service manager rather than to a session. The rest of the run went
through that path: unit `p09-driver`, `run1/bin/p09_launch_unit.sh`, and
`run1/bin/p09_resume.sh`, which waits for any orphaned node step, stages
finished points, then runs the driver.

Two further fixes:

- Spur's `srun` intermittently never connects: about half of the calls
  during this window hung with no output. `on_node` now has each remote command
  echo a marker first, and retries when no marker arrives within 90 s.
- A retry that connected late could measure a point twice. Each A/B point now
  runs under a node-side `flock` and skips if the node already holds a `DONE`
  copy. That check fired once, on `ab_r2_c32_a0`, when the login node's NFS view
  still lagged the staged copy.

The five-minute guard was removed. It treated a timed-out check as "node idle"
and relaunched the driver into a running point. Nothing was lost only because
the relaunched driver died before it reached the node.

**Round 2** (same node, same seed, arm order reversed):

| conc | knob | tok/s/GPU | p50 TPOT ms | p90 TPOT ms | p90 TTFT s | KV hit | reqs | duration s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 289.868 | 10.14 | 16.93 | 1.527 | 0.9032 | 1043 | 616.4 |
| 32 | 0 | 298.336 | 9.44 | 16.71 | 1.714 | 0.9052 | 1059 | 629.1 |
| 8 | 1 | 76.403 | 3.86 | 4.59 | 0.752 | 0.9177 | 227 | 600.7 |
| 8 | 0 | 76.199 | 3.87 | 4.62 | 0.813 | 0.9182 | 225 | 601.0 |

Knob on / off, per pair and mean:

| conc | round 1 | round 2 | mean |
| ---: | ---: | ---: | ---: |
| 32 | 0.9574 | 0.9716 | **0.9645** |
| 8 | 1.0026 | 1.0027 | **1.0026** |

The two knob-off C32 replays agree to 0.02% (298.390 and 298.336), and the
knob-on pair to 1.5%, so the C32 loss is well outside this node's noise. It is
in decode: p50 TPOT rises 0.70–0.85 ms in both pairs. The C8 gain is
consistent across both pairs but small. One replay sits outside the usual C32
duration band of 620–632 s: round 2's knob-on replay ran 616.4 s. If that is a
short-run artifact, it inflates the knob-on number, so the true C32 loss is at
least as large as measured. Seed 42 only, which the §5 rule allows for a
rejection.

**Verdict: reject PR #19's FlyDSL MoE config set as merged.** It costs 3.5% at
C32 for 0.26% at C8. That is not a Pareto improvement, and it trips §6 ("the
kernel wins but end-to-end loses at either point"). Implications:

- The plan branch's knob defaults to on. It should default to off until a
  reduced set passes.
- The base `arbor-v2-plus-v6best-pr19 @ be794db46` carries PR #19's own
  install, which is not what either arm measured: it writes into the AITER
  package and replaces the table with only its 16 rows. So the base itself is
  unmeasured, and the knob-on arm is the closest measured proxy for it.
- Next candidate, per §6 and the trace comparison: the C8 verify tier (64) is
  neutral-to-positive and the C32 verify tier (256) is where decode loses.
  Measure the same configs without tier 256 and without the `_xcd4` stage-2
  kernels, paired the same way, before touching prefill tiers.

### 2026-09-25: follow-up, reduced tier set

**Candidate.** Plan branch `2eb10620e` adds `VLLM_DSV41_FLYDSL_MOE_TIERS`, a
comma-separated list of the tiers to install. Empty installs all 16 rows, and
the CSV hash stays `9f41779ed160`. An unknown tier raises. Eight of the 16
tiers (32–2048) use the `_xcd4` stage-2 kernel, including both verify tiers,
64 (C8) and 256 (C32). The reduced set drops all eight and keeps
`1,2,4,8,16,4096,8192,16384`: the small-M rows plus the large-prefill tiers
that the trace showed 9–31% faster per call.

Probes in the container before the run:

- full set: 16 rows;
- reduced set: 8 rows, CSV hash `08db1f46175b`;
- unknown tier: raises;
- knob 0: 0 rows.

Each arm's marker pattern was checked offline and matches only its own arm.
The run used `run2/`, the same node and seed, and the harness from round 2,
with one fresh knob-off C32 control as a drift check.

| conc | arm | round | tok/s/GPU | p50 TPOT ms | p90 TPOT ms | KV hit | reqs | duration s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | reduced 8 | 3 | 299.125 | 9.51 | 16.95 | 0.9056 | 1064 | 627.9 |
| 32 | off | 3 | 297.904 | 9.36 | 16.93 | 0.9047 | 1058 | 628.1 |
| 8 | reduced 8 | 3 | 76.446 | 3.87 | 4.63 | 0.9183 | 227 | 600.4 |
| 32 | reduced 8 | 4 | 298.252 | 9.42 | 17.33 | 0.9047 | 1058 | 628.9 |

Against this node's control means (C32: three readings, 298.210, spread 0.16%;
C8: two readings, 76.233, spread 0.09%):

| conc | all 16 tiers | reduced 8 tiers |
| ---: | ---: | ---: |
| 32 | 0.9650 (r1 0.9580, r2 0.9720) | **1.0016** (r3 1.0031, r4 1.0001) |
| 8 | 1.0026 (r1 1.0031, r2 1.0022) | **1.0028** (r3 only) |

Round 4's C8 point was lost. Job 174441 was cancelled at about 16:00 UTC, while
that point was booting, and `m2m-016` went to a new job at the same time. The
driver wrote `STATUS` DONE regardless; the point reads `NO_STATUS`.

**Result.** The reduced set removes the whole C32 loss and keeps C8's small
gain. So the loss came from the `_xcd4` tiers, and C32 decode sits in one of
them (tier 256). The C8 gain survives without tier 64, so it does not come from
the C8 verify tier; the remaining candidates are tiers 1–16 or the prefill
tiers. Neither point improves beyond noise: C32 is +0.16% against a 0.16%
spread, and C8 is +0.28% on one reading. Seed 42 only.

**Decision.** The reduced set is safe to carry, and it replaces the full set as
the knob-on configuration. It is not a throughput win, and Plan 09's goal of
lowering MoE time is not met end to end. Before calling it a gain it would need
the missing C8 point, three seeds (§5), and a trace showing where the C8 gain
comes from.
