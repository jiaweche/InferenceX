# DeepSeek V4.1 Flash MegaMoE Ruby Handoff

> Branch: `dsv41-flash-megamoe-agentx`
>
> Handoff date: 2026-09-15
>
> Goal: continue the DeepSeek-V4.1-Flash MegaMoE investigation directly on
> Ruby without repeating model staging or confusing operator evidence with an
> end-to-end serving result.

## 1. Bottom line

Ruby is the easiest place to continue because the model, pinned container,
AITER/MoRI source builds, JIT cache, and raw benchmark evidence are under the
same node-local `/scratch` tree.

The investigation found credible, shape-dependent communication-fused Stage2
speedups for the exact V4.1 TP4 physical expert shape. It did **not** validate
an end-to-end MegaMoEV2 improvement:

- the tested operator was AITER `comm_fused_moe` (GEMM2 + TP collective), not
  the plan's full expert-parallel MegaMoEV2 dispatch/GEMM/combine path;
- the candidate is not integrated into the pinned vLLM image;
- the independent distributed correctness oracle fails in its ordinary arm;
- all measured candidate runs segfault during MoRI communicator teardown;
- uniform routing at M=16 regresses and therefore needs ordinary fallback.

Treat the measurements as evidence to continue, not as a performance claim.

## 2. Allocation and host state

The previous validation allocation, job `37193`, was released after evidence
was copied to persistent storage.

A continuation allocation has been submitted:

```text
job:       37515
name:      dsv41-megamoe
partition: meta64
limit:     2-00:00:00
node:      cv350-rck-g03-c09-18
```

It deliberately targets the same node so the node-local model and build
artifacts can be reused. At handoff creation it was pending with
`Reason=Priority`; do not submit a duplicate allocation.

From a laptop:

```bash
ssh ruby-1 'squeue -j 37515 -o "%.18i %.28j %.2t %.10M %.12l %R"'
```

When it reaches `R`:

```bash
ssh cv350-rck-g03-c09-18
```

Before any GPU run, verify that the allocation still belongs to this user and
that all GPUs are idle:

```bash
squeue -h -u "$USER" -w "$(hostname -s)"
rocm-smi --showuse --showmemuse
docker ps
```

Release the job when work is finished:

```bash
ssh ruby-1 'scancel 37515'
```

Ruby SSH was intermittently unavailable during the previous run, while Slurm
and the detached benchmark continued correctly. Launch long runs detached and
write logs/results to disk rather than depending on one SSH connection.

## 3. Repository and data locations

Node-local, ephemeral paths on `cv350-rck-g03-c09-18`:

```text
/scratch/jiaweche/inferencex-dsv41flash/
  repo/                         InferenceX worktree used for validation
  models/DeepSeek-V4.1-Flash/   476 GB pinned checkpoint
  hf-cache/
  hf-home/
  runtime/
  runs/

/scratch/jiaweche/dsv41-megamoe/
  aiter/                        AITER source for the SGLang image
  aiter-vllm/                   clean AITER source/JIT build for the vLLM image
  mori/                         MoRI source
  pydeps-vllm/                  Python 3.12 MoRI build used with vLLM
  results/                      raw operator logs and aggregate JSON
```

Persistent evidence copied to NFS:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/
  agentx/smoke/
  agentx/fast-baseline/
  operator/
```

The persistent tree is the source of truth if scratch has been reclaimed.
Do not write large models or build caches under `/home/jiaweche`; only 100 GB
is available there.

## 4. Pinned provenance

```text
worker:          cv350-rck-g03-c09-18
GPU ISA:         gfx950, device 0x75a0
host amdgpu:     6.14.14

model:           deepseek-ai/DeepSeek-V4.1-Flash
model revision:  dba1be0a40aa45a94ad051997016db3960a90277

vLLM image:      vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657
image digest:    sha256:960228cfcb5de9f4cd22d28998d1125be62b546c3d220a884f570343e99ffcee

InferenceX base: 730bc4edf46010d76804ef4282419b1d174beb56
AITER source:    797cce253bbadbaf651cdd93527806aa989ea56b
MoRI source:     0a1cd437317463f96906f20ca0566e4e969d8471
```

The pinned vLLM image itself contains neither the tested
`CommFusedMoeRuntime` nor `mori`. The operator probe mounted the newer AITER
source and a locally built MoRI Python 3.12 package into that image.

## 5. Branch changes completed

The branch contains:

- `plans/DSV41_FLASH_MEGAMOE_AGENTX.md`: the full bring-up and integration
  plan;
- `runners/run_dsv41flash_agentx_mi350x-ruby.sh`: a dedicated V4.1 Ruby
  vLLM/DSpark preset, separate from the V4-Pro SGLang preset;
- `runners/launch_mi350x-ruby.sh`: model-revision propagation and provenance;
- `benchmarks/single_node/agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh`: exact
  checkpoint revision support;
- this handoff.

The dedicated preset defaults to a 1200-second fast baseline. Set
`RUBY_DSV41FLASH_SMOKE=1` for the 60-second smoke.

## 6. Real-model baseline results

### Startup smoke

```text
topology:             TP4 / EP1
AgentX concurrency:   1
duration:             60 seconds
profiled requests:    5
request errors:       0
output throughput:    21.41 tokens/s/GPU
```

This proves model load, graph capture, DSpark, AgentX replay, metrics export,
and clean shutdown. It is not a performance result.

### Fast C8 ordinary baseline

```text
topology:                     TP4 / EP1
AgentX concurrency:           8
profile duration:             1214.26 seconds
profiled requests:            376
request errors:               0
P90 E2E normalized intvty:    90.12903
P90 TTFT:                     1.03791 seconds
P90 TPOT:                     7.87 ms
total output throughput:      289.25282 tokens/s
output throughput per GPU:    72.31320 tokens/s/GPU
```

Primary JSON:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/agentx/fast-baseline/
  dsv41flash_tp4_c8_fast_baseline_20260915T160255Z.json
```

The server log confirms that ordinary AITER uses the V4.1 TP4 physical shape:

```text
model_dim=5120
local padded inter_dim=640
experts=384
topk=6
activation=FP8
weight=MXFP4
```

## 7. Operator probe and results

The probe compared:

```text
ordinary: AITER A8W4 Stage2 + TP all-reduce
candidate: AITER comm_fused_moe Stage2/TP megakernel
```

Each cell below is the median speedup from three serial repetitions:

| M | uniform route | skewed route |
| ---: | ---: | ---: |
| 1 | 1.452x | 1.370x |
| 2 | 1.106x | 1.346x |
| 4 | 1.080x | 1.332x |
| 8 | 1.011x | 1.153x |
| 16 | **0.769x** | 1.119x |

Detailed aggregate:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/operator/
  combined-v41-tp4-smallm-summary.json
```

Candidate-relative numerical results stayed inside the tuner gate:

```text
maximum absolute error: 0.171875  (limit 1.0)
maximum relative L2:    0.021625  (limit 0.05)
```

These values compare the candidate against the ordinary AITER control. They do
not supersede the failed independent Torch oracle described below.

## 8. Failures that must be fixed

### 8.1 Independent correctness oracle

The upstream distributed test fails before evaluating the candidate because
its ordinary eager arm disagrees with its Torch reference:

```text
max_abs=5.437500
rel_l2=0.417882
err=0.365792
```

The exact single-GPU ordinary A8W4 MoE case completed result collection, so the
failure appears specific to the distributed Stage2/reference contract. Do not
weaken tolerances or promote candidate-relative results as a substitute.

Logs:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/operator/
  upstream-tp8-comm-ut.log
  upstream-tp8-comm-ut-vllm.log
```

### 8.2 MoRI teardown

All 30 measured candidate repetitions emitted complete timing and numerical
results, then every rank received SIGSEGV during communicator cleanup.

This is a hard stability failure. Isolate `mori.cco.Communicator` construction,
window registration, explicit destruction, and process-group teardown before
attempting a serving integration.

### 8.3 Selective dispatch requirement

Uniform M=16 was about 23% worse in speedup terms (`0.769x`). Any integration
must be opt-in, bucket-selective, and fail closed to ordinary AITER. Do not
enable the candidate globally.

### 8.4 Integration gap

The current V4.1 vLLM model leaves `use_mega_moe` disabled and
`finalize_mega_moe_weights()` as a no-op. There is no merged adapter that owns:

- weight/scale conversion;
- MoRI lifecycle and symmetric windows;
- EP group initialization;
- DSpark fallback;
- graph warmup/capture;
- zero-token ranks;
- bucket-selective fallback.

## 9. Recommended continuation order

1. **Confirm job 37515 and scratch retention.**
   Check the checkpoint index, image digest, AITER/MoRI commits, and persistent
   evidence before downloading or rebuilding anything.
2. **Separate the two candidate concepts.**
   Keep the measured TP Stage2 `comm_fused_moe` evidence distinct from the
   plan's full EP MegaMoEV2 candidate. Decide which implementation is being
   productized before editing vLLM.
3. **Fix the hard operator gates.**
   Reproduce and root-cause the ordinary distributed oracle failure and the
   MoRI teardown SIGSEGV. Require clean exit on all ranks.
4. **Capture real AgentX route frequencies.**
   Capture backbone and DSpark M buckets, routing skew, graph buckets, and
   GPU-time weights. The current matrix proves route sensitivity but cannot
   weight expected end-to-end gain.
5. **Prove the selected candidate independently.**
   Cover zero-token ranks, uneven rank counts, uniform/skew routes, graph
   replay, repeated bursts, and maximum configured capacity.
6. **Add an explicit vLLM adapter.**
   Build a pinned candidate image. Keep ordinary AITER as the default and
   fallback.
7. **Run an equal-topology A/B.**
   For full MegaMoEV2, compare TP4/EP4 MoRI + ordinary AITER against TP4/EP4
   MoRI + MegaMoEV2. Do not compare pure TP4 directly to EP4 and attribute the
   difference to the kernel.
8. **Promote only after matched AgentX and accuracy gates.**
   Use three serial 1200-second C8 runs per arm, then one matched 3600-second
   confirmation and the real DSpark accuracy run.

## 10. Reusing the V4.1 baseline

After job 37515 starts:

```bash
ssh cv350-rck-g03-c09-18
cd /scratch/jiaweche/inferencex-dsv41flash/repo
git status --short --branch
```

Verify the checkpoint:

```bash
test -s \
  /scratch/jiaweche/inferencex-dsv41flash/models/DeepSeek-V4.1-Flash/model.safetensors.index.json
```

Smoke:

```bash
export GITHUB_WORKSPACE=/scratch/jiaweche/inferencex-dsv41flash/repo
export RUBY_DSV41FLASH_SMOKE=1
export TP=4 EP_SIZE=1 CONC=1
bash runners/run_dsv41flash_agentx_mi350x-ruby.sh
```

Fast ordinary C8:

```bash
export GITHUB_WORKSPACE=/scratch/jiaweche/inferencex-dsv41flash/repo
export RUBY_DSV41FLASH_SMOKE=0
export TP=4 EP_SIZE=1 CONC=8 DURATION=1200
export AIPERF_WARMUP_REQUESTS_PER_LANE=1
export AIPERF_EXPERIMENTAL_FAST=1
bash runners/run_dsv41flash_agentx_mi350x-ruby.sh
```

Do not rerun the baseline merely to confirm that files exist. Start from the
operator correctness and teardown blockers unless the environment provenance
has changed.

## 11. Continuation results on e18-08

This section records the September 15 continuation after the original handoff.
It does not change the distinction between the measured TP Stage2
`comm_fused_moe` path and the full EP MegaMoEV2 candidate.

### 11.1 Allocation and restored provenance

Job `37535` provided a temporary allocation on
`cv350-rck-g03-e18-08`. The original job `37515` later also started on
`cv350-rck-g03-c09-18`; do not leave both allocations running unnecessarily.

The e18-08 worker matched the required hardware and software provenance:

```text
GPU:              gfx950, device 0x75a0
host amdgpu:      6.14.14
vLLM image ID:    sha256:960228cfcb5de9f4cd22d28998d1125be62b546c3d220a884f570343e99ffcee
InferenceX:       77f38a2c51f6ee45202754a0e2e6f298c9915da5
AITER base:       797cce253bbadbaf651cdd93527806aa989ea56b
MoRI base:        0a1cd437317463f96906f20ca0566e4e969d8471
model revision:   dba1be0a40aa45a94ad051997016db3960a90277
checkpoint:       48/48 shards, 476 GB
```

### 11.2 Distributed oracle and runtime ABI fixes

The ordinary-control mismatch was a real test ABI error. The tuned ordinary
`flydsl_moe2_layout_*` Stage2 kernel consumes a padded, expert-sorted
intermediate. The communication-fused producer resolves sorted route IDs back
to compact `[token, slot, K]` rows. The test had passed the compact buffer to
both arms.

The AITER patch now:

- supplies the sorted buffer to ordinary Stage2 and the compact buffer to the
  communication-fused arm and Torch oracle;
- requests compact Stage1 output when `CommFusedMoeRuntime` selects a producer
  that resolves token-major rows;
- retains the ordinary layout for the atomic wrapper;
- closes every runner deterministically before process-group teardown.

The final TP8 command completed with exit code zero:

```bash
torchrun --standalone --nproc_per_node=8 \
  op_tests/multigpu_tests/test_comm_fused_moe.py \
  --tokens 1 --routes uniform --graph-replays 1
```

Ordinary, communication-fused, runtime-padding, separate-stream, and graph
checks all passed. The M=1 Stage2 maximum errors were `0.0625` ordinary and
`0.21875` communication-fused. The M=3 runtime maximum errors were `0.71875`
ordinary and `0.9375` communication-fused, inside the unchanged gates.

### 11.3 MoRI teardown root cause and fix

A native GDB trace reduced the failure to
`c10d::symmetric_memory::AllocationRef::~AllocationRef`. Both CCO and PyTorch
called `hipMemRelease` on the same handle value returned by
`hipMemRetainAllocationHandle`. ROCm 7.2.3 exposes the external allocator's
handle here; it is not a separately releasable CCO-owned handle.

The minimal MoRI patch adds ownership to `ccoComm::AllocMeta`, marks external
imports borrowed, unmaps their CCO alias, and leaves the physical-handle
release to the external owner. CCO allocations remain owned and released.

Validated clean exits:

- one-rank external-window create/register/close/destroy;
- four-rank external-window create/register/close/destroy;
- the exact TP4 V4.1 tuner;
- the TP8 independent oracle/runtime test.

The installed validation library SHA256 was:

```text
3df6da1342f1c9dc7923fd2620bb132b283b2063bf0040a88cb08056e136cfd5
```

### 11.4 Exact 60-second AgentX shape capture

The capture used C8, TP4/EP1, DSpark, the pinned checkpoint, and a
diagnostic-only image derived from the pinned vLLM image. `--enforce-eager`
was required so Python observed every routing call. The capture gate opened
only for the 60-second profiling interval. This run is not performance
evidence.

The exact window contains 17,255 MoE events. Every route ID was in range and
every per-expert vector summed to `raw_m * topk`.

Backbone:

- 99.25% of calls were decode `M=6`, with Mega padding bucket 8.
- Route-work fractions were 18.38% decode M=6, 6.83% prefill M=888,
  20.19% prefill M=2625, and 54.60% prefill M=7100.
- Mean hottest-expert shares were 77.2% for decode and 45-47% for prefill.
- Some layers routed 97.8-98.2% of prefill tokens to one expert.

DSpark:

- every observed call was `M=5`, top-k 3, 128 experts, Mega padding bucket 8;
- the mean hottest-expert share was 84.7%, reaching 100%;
- only 7.13 experts were active on average across 15 route slots.

Raw and summarized evidence:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/agentx/shape-capture-v3/
  moe_shape_capture_60s.jsonl
  moe_shape_summary_60s.json
  routes/
  routes-count-exact/
```

### 11.5 Full EP4 MegaMoEV2 gates

These tests are full EP MegaMoEV2, not the earlier TP Stage2
`comm_fused_moe` measurements.

With `max_tok_per_rank=16384`, the compact path passed:

- captured backbone M=6 versus the independent Torch oracle:
  `relL2=0.058207`, graph replay exact;
- captured DSpark M=5 versus the independent Torch oracle:
  `relL2=0.057710`, graph replay exact;
- captured prefill M=888, 2625, and 7100: graph replay exact;
- captured prefill M=888 versus the independent Torch oracle:
  `relL2=0.058260`;
- uneven per-rank M=`[0,1,6,3]`, including a zero-token rank:
  `relL2=0.057873`, graph replay exact;
- 20 repeated captured-M=6 bursts;
- the maximum configured MTPR case M=16384, graph replay exact.

Preliminary equal-topology EP4 operator A/B used MoRI dispatch/combine plus
ordinary AITER as control and MegaMoEV2 as candidate. Rank-maximum latency:

```text
captured backbone M=6:   0.0979 ms control, 0.1925 ms candidate, -49.1%
captured DSpark M=5:    0.0851 ms control, 0.1408 ms candidate, -39.6%
captured prefill M=888: 1.8430 ms control, 0.8488 ms candidate, +117.2%
captured prefill M=2625:2.8739 ms control, 1.8782 ms candidate, +53.0%
captured prefill M=7100:5.5367 ms control, 4.5179 ms candidate, +22.6%
```

A count-exact reconstruction of the measured layer-10 M=7100 distribution
(98.2% hottest-expert share; reconstructed IDs, not raw IDs) also favored
MegaMoEV2: `6.3574 ms` control versus `5.0179 ms` candidate, `+26.7%`.

The critical negative and positive points were repeated three times:

```text
captured backbone M=6 median:
  0.0979 ms control, 0.1925 ms candidate, -49.12%

count-exact layer-10 M=7100 median:
  6.3603 ms control, 5.0242 ms candidate, +26.59%
```

All six runs exited cleanly. The observed speedup ranges were
`-49.59%` to `-48.89%` for M=6 and `+26.39%` to `+26.69%` for the
count-exact M=7100 route.

The current evidence supports only a phase- and bucket-selective prototype:

- keep ordinary fallback for backbone decode M=6, DSpark M=5, the previously
  regressing uniform M=16 TP Stage2 case, mixed phases, and unknown buckets;
- consider MegaMoEV2 only for proven backbone prefill buckets;
- require the same decision on every EP rank.

Do not infer an end-to-end gain. The intermediate-bucket A/B values remain
single operator measurements, M=888 is the only large bucket checked against
the independent Torch oracle, and no vLLM EP adapter or matched serving A/B
exists yet.

### 11.6 Checked-in continuation artifacts

The branch carries reproducible, base-pinned overlays and analysis tooling:

```text
experimental/dsv41_megamoe/
  Dockerfile.shape_capture
  vllm-moe-shape-capture.patch
  aiter-comm-fused-correctness-lifecycle.patch
  mori-external-window-borrowed-handle.patch
  analyze_shape_capture.py
  test_analyze_shape_capture.py
runners/test_dsv41flash_megamoe_capture.py
```

Remaining gates before a serving adapter:

1. Repeat the intermediate M=888 and M=2625 operator A/B points if they become
   part of the proposed selective policy.
2. Extend independent numerical coverage beyond M=888 if a larger prefill
   bucket enters the proposed selective policy.
3. Define phase detection and rank-consistent fallback for mixed batches.
4. Build reviewed AITER, MoRI, and vLLM branches into one immutable image.
5. Only then run matched EP4 control/candidate AgentX and accuracy tests.

