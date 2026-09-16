# DeepSeek V4.1 Flash MegaMoE + AgentX Plan

> Working implementation plan for branch `dsv41-flash-megamoe-agentx`.
> This is an internal execution document, not published InferenceX documentation.

## 1. Objective

Bring up `deepseek-ai/DeepSeek-V4.1-Flash` on Ruby MI350X using the validated
InferenceX vLLM/ROCm AgentX path, capture the real MoE workload distribution,
integrate AITER MegaMoEV2 behind an explicit opt-in, and retain the change only
if matched AgentX and accuracy gates pass.

The end-to-end objective is AgentX serving performance. Kernel or
microbenchmark speedups are necessary evidence, but they are not sufficient to
ship the integration.

### 1.1 Execution plan index

This file is the master plan. Keep focused implementation details in linked
child plans and roll their final revisions, evidence, and decisions back into
this file.

Completed child plan:

- [Pooled AITER workspace and explicit teardown](./03_DSV41_FLASH_MEGAMOE_POOLED_AITER_WORKSPACE.md)

Executed child plan, stopped before long A/B:

- [Selective vLLM MegaMoEV2 adapter](./04_DSV41_FLASH_MEGAMOE_VLLM_ADAPTER.md)

Validated continuation evidence:

- [Ruby handoff and measured shape/A-B results](./02_DSV41_FLASH_MEGAMOE_RUBY_HANDOFF.md)

The vLLM adapter child completed implementation, TP4/EP4/DSpark path
attribution, graph fallback, and the directional matched smoke. The two-request
hybrid comparison missed the regression gate, so plan 05 was not created and
the three 1200-second repetitions were not started. Follow-up profiling found
that Mega covered only 1.86% of observed non-capture layer calls; dominant
M≈16K prefill chunks remained above the MTPR=8192 workspace limit.

## 2. Branch and source baseline

The feature branch is:

```text
dsv41-flash-megamoe-agentx
```

It was created from `ruby-1-dsv4-agentx-bringup`, then rebased onto:

```text
upstream/main @ b3c2f1efa
```

The rebase retained the reusable Ruby commits:

```text
e0456b3cf  feat(runners): add Ruby MI350X Docker launcher
730bc4edf  feat(runners): add Ruby DeepSeek V4 AgentX preset
```

The old `ruby-1-dsv4-agentx-bringup` branch remains untouched at
`0b1da44a`.

The existing `run_dsv4_agentx_mi350x-ruby.sh` preset is a pattern only. It
targets DeepSeek V4 Pro through SGLang and must not be modified in place for
V4.1 Flash. V4.1 needs a separate vLLM preset.

## 3. Verified facts

### 3.1 Model identity

Pin the public checkpoint revision:

```text
deepseek-ai/DeepSeek-V4.1-Flash
revision dba1be0a40aa45a94ad051997016db3960a90277
```

The checkpoint occupies approximately 510 GB of storage. The official model
card describes a 552B-parameter backbone plus 196B Engram memory.

Relevant text-model geometry:

| Property | Value |
| --- | ---: |
| Text layers | 40 |
| Hidden size | 5120 |
| Routed expert intermediate size | 2304 |
| Routed experts | 384 |
| Shared experts | 1 |
| Routed experts selected per token | 6 |
| Expert weight type | FP4 |
| Activation quantization | Dynamic FP8 |
| Expert block scale | UE8M0, 1x32 |
| Maximum context | 1,048,576 |
| DSpark layers | 3 |
| DSpark routed experts | 128 |
| DSpark experts selected per token | 3 |

The routed expert GEMM shapes before packing are:

```text
GEMM1: M x 5120  @  5120 x 4608
GEMM2: M x 2304  @  2304 x 5120
```

`M` is dynamic and must come from the real AgentX trace rather than an assumed
synthetic batch size.

### 3.2 Validated MI355X baseline

InferenceX already contains the validated config:

```text
dsv41flash-fp4-mi355x-vllm-agentic-dspark
```

Authoritative files:

- `configs/amd-master.yaml`
- `benchmarks/single_node/agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh`
- `golden_al_distribution/dsv41flash_dspark.yaml`

Pinned runtime:

```text
image: vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657
model: deepseek-ai/DeepSeek-V4.1-Flash
framework: vllm
topology: TP4
KV offload: none
AgentX concurrency: 1, 2, 4, 8, 16, 32
```

Upstream run `34710937012` passed the MI355X concurrency 1-32 sweep and the
eval-only concurrency-32 job.

Important recipe behavior:

- `VLLM_ROCM_USE_AITER=1`
- `VLLM_ROCM_USE_AITER_MOE=1`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- `--moe-backend aiter`
- `--language-model-only`
- `--max-num-seqs 128`
- `--max-num-batched-tokens 16384`
- full-context AgentX corpus `semianalysis_cc_traces_weka_062126`
- DSpark with five draft tokens
- synthetic acceptance length 3.51 for throughput
- real block rejection for accuracy
- adaptive verification disabled on ROCm

The validated backend is ordinary AITER CK A8W4 fused expert compute. It does
not fuse expert-parallel dispatch and combine communication.

### 3.3 MegaMoE status

AITER MegaMoEV2 was merged in ROCm/aiter PR 4439 and is available in AITER
v0.1.22. It:

- supports A8W4 only;
- fuses dispatch/sort/GEMM1 and GEMM2/weighted P2P combine;
- uses symmetric MoRI memory;
- requires synchronized expert-parallel ranks;
- requires a power-of-two maximum-token-per-rank capacity (MTPR);
- supports world sizes 1-8 and top-k 1-16.

Its current public tests cover DeepSeek V4 Pro, GLM-5.2, and Kimi routing, not
DeepSeek V4.1 Flash.

The V4.1 tuple is source-shape-admissible:

- `5120` is divisible by 256;
- `2304` is divisible by 256;
- `2 * 2304` is divisible by the selected 512-wide Stage-1 tile;
- EP4 has 96 experts/rank and stays under the 256-expert/rank limit;
- EP8 has 48 experts/rank, matching the optimized V4 Pro reference layout;
- top-k 6 is supported.

This proves only that the visible API constraints accept the shape. It does not
prove real checkpoint weight conversion, correctness, graph replay, or a
performance gain.

No merged vLLM adapter currently invokes AITER MegaMoEV2. The ROCm V4.1 model
explicitly leaves `use_mega_moe` disabled and its
`finalize_mega_moe_weights()` hook is a no-op.

Open SGLang integrations (PRs 35619 and 36269) are design references, not
merge-ready dependencies for this vLLM plan.

### 3.4 Similar names that are not this implementation

- AITER/vLLM FHMoE fuses shared-expert and routed-expert compute. It does not
  fuse EP communication and is explicitly tuned for V4 Pro geometry.
- NVIDIA DeepGEMM MegaMoE is a CUDA implementation and is not portable to
  gfx950.
- Primus-Turbo MegaMoE is primarily a training/autograd implementation. Its
  existing DeepSeek V3 BF16 kernels and measurements are useful design
  evidence, but they are not a drop-in inference backend.

The first serving implementation will therefore use AITER MegaMoEV2. If the
project requirement changes to require Primus-Turbo ownership specifically,
the bring-up and shape-capture phase remains valid, but the Phase 2 kernel
implementation repository changes.

## 4. Phase 1: Bring up DeepSeek V4.1 Flash

### 4.1 Add a dedicated Ruby preset

Create:

```text
runners/run_dsv41flash_agentx_mi350x-ruby.sh
```

It should wrap `runners/launch_mi350x-ruby.sh` and set:

```text
IMAGE=vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
MODEL_PREFIX=dsv41flash
FRAMEWORK=vllm
PRECISION=fp4
SPEC_DECODING=mtp
TP=4
KV_OFFLOADING=none
TOTAL_CPU_DRAM_GB=0
BENCHMARK_GPU_LABEL=mi355x
WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
RUBY_SCRATCH_ROOT=/scratch/$USER/inferencex-dsv41flash
```

The result metadata must identify the physical system as Ruby MI350X even
though the shared gfx950 benchmark filename contains `mi355x`.

Do not replace the existing DeepSeek V4 Pro preset.

### 4.2 Harden the generic Ruby launcher only where required

Keep the existing scratch-backed mounts for:

- repository;
- model;
- Hugging Face cache;
- AIPerf mmap cache;
- runtime files;
- result artifacts.

Before changing the launcher, dry-run V4.1 script resolution:

```text
EXP_NAME=dsv41flash_tp4_...
FRAMEWORK=vllm
SPEC_DECODING=mtp
```

It must resolve to:

```text
benchmarks/single_node/agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh
```

Possible launcher changes, only if the smoke proves they are necessary:

- `/ix` mount parity with the official MI355X launcher;
- vLLM/MegaMoE environment pass-through;
- a pre-launch VRAM drain gate;
- exact model revision recording;
- installed vLLM/AITER/MoRI commit and version recording.

### 4.3 Local validation before GPU work

Run:

1. Bash syntax for the new preset, generic Ruby launcher, and V4.1 benchmark.
2. YAML parse for `configs/amd-master.yaml`, `configs/runners.yaml`, and
   `perf-changelog.yaml`.
3. Exact-key matrix generation for
   `dsv41flash-fp4-mi355x-vllm-agentic-dspark`.
4. Script-resolution tests for TP4, vLLM, and `_mtp`.
5. Existing focused runner and matrix tests.
6. `git diff --check`.

Inspect the generated values rather than accepting only a zero exit code:

- image and model;
- runner;
- TP/EP/DP/DCP/PCP;
- concurrency;
- KV offload;
- DSpark mode;
- duration;
- eval flags.

### 4.4 Runtime preflight when Ruby returns

On an allocated `cv350-rck-*` worker:

1. Confirm MI350X/gfx950 identity and clean VRAM.
2. Confirm at least 510 GB scratch capacity plus image/runtime headroom.
3. Pull and digest-pin the exact vLLM image.
4. Stage the checkpoint at its exact HF revision.
5. Initialize the pinned AIPerf submodule.
6. Record:
   - host driver;
   - ROCm version;
   - vLLM commit;
   - AITER version and commit;
   - MoRI version and commit;
   - `MegaMoEV2` import availability;
   - image ID/digest;
   - model revision.

Do not start MegaMoE work until the ordinary AITER baseline is reproducible.

### 4.5 Bring-up run sequence

#### B0: startup smoke

```text
TP4, concurrency 1, 60 seconds, one warmup request/lane
```

This run uses the unsafe-duration override and is not publishable. It proves
only model load, server readiness, AgentX connectivity, metrics, and artifact
generation.

#### B1: fast baseline

```text
TP4, concurrency 8, 1200-second AgentX fast mode
```

Use the exact upstream model/image/DSpark settings. This is the primary
iteration baseline for shape capture and future A/B work.

#### B2: accuracy baseline

Run eval-only with real DSpark block rejection and adaptive verification
disabled. Preserve raw results and samples. The upstream configured GSM8K
acceptance floor is 0.90.

#### B3: canonical baseline

```text
TP4, concurrency 8, 3600 seconds
```

Preserve:

- aggregate JSON;
- AIPerf raw artifacts;
- server metrics;
- server log;
- launch metadata;
- exact serve and benchmark commands;
- GPU telemetry;
- image and source provenance.

After the target path is stable, widen to the upstream concurrency matrix
`[1, 2, 4, 8, 16, 32]`.

## 5. Phase 1B: Capture real MoE shapes

Instrument the ordinary AITER path before authoring or tuning MegaMoE.

Each record must include:

| Field | Purpose |
| --- | --- |
| model revision and layer | Reproducibility |
| backbone vs DSpark layer | Separate 384/top-6 from 128/top-3 |
| prefill, decode, or draft/verify | Different token distributions |
| local and global token count | MegaMoE MTPR and dispatch sizing |
| padded/token-bucket M | Kernel selection |
| GEMM leg and M/N/K | Microbenchmark construction |
| activation/weight/scale dtype | A8W4 contract |
| EP world size and experts/rank | Dispatch geometry |
| top-k route IDs and weights | Correctness |
| route count per rank/expert | Skew and hot-expert behavior |
| selected backend/kernel | Confirm actual dispatch |
| CUDA-graph bucket/capture state | Graph fidelity |
| kernel and whole-step latency | Attribution |

Collect at least:

```text
TP4 baseline: concurrency 1, 8, 16, 32
```

The output is a versioned shape manifest. All subsequent microbenchmark
weights and token buckets must be derived from it.

## 6. Phase 2A: Prove MegaMoEV2 independently

### 6.1 Extend AITER tests and microbenchmarks

Add a V4.1 network definition to the AITER MegaMoEV2 tests:

```text
model_dim=5120
inter_dim=2304
experts=384
topk=6
swiglu_limit=10.0
```

Also test the DSpark routing shape:

```text
model_dim=5120
inter_dim=2304
experts=128
topk=3
```

Exercise:

- EP4 and EP8;
- every dominant AgentX token bucket;
- zero-token ranks;
- uneven rank token counts;
- uniform routing;
- hot-expert skew;
- cross-rank fanout boundaries;
- eager execution;
- CUDA-graph capture and replay;
- repeated burst execution;
- maximum configured MTPR.

### 6.2 Use an equal-topology control

Do not compare the current pure-TP server directly with MegaMoE. That would
confound topology and kernel effects.

The first controlled operator/server A/B is:

```text
Control:
  expert parallel enabled
  MoRI dispatch/combine
  AITER CK A8W4 experts

Candidate:
  same expert-parallel topology
  AITER MegaMoEV2 fused dispatch/GEMM/combine
```

Preferred initial topology:

```text
TP4 / EP4 on four GPUs
```

Add EP8 as a separate experiment if the microbenchmark supports it. EP8 has
48 experts/rank and matches AITER's optimized reference geometry, but it uses
twice the GPUs and must be compared only against an EP8 control.

### 6.3 MTPR constraint

An aggregated server handles prefill and decode with one MegaMoE instance.
With `--max-num-batched-tokens 16384`, its safe MTPR is at least 16384.
Therefore it uses MegaMoEV2's compact path, not the optimized low-latency
fixed-slot path.

Do not reduce MTPR below the reachable prefill bound merely to obtain a faster
decode kernel.

If the compact path does not win, the next architecture is prefill/decode
disaggregation:

```text
prefill MTPR: >= effective prefill chunk bound
decode MTPR:  <= 128 where proven safe
```

That is a follow-up architecture, not part of the first integration.

### 6.4 Microbenchmark gates

Required correctness:

- finite output on all ranks;
- accepted A8W4 numerical tolerance against the ordinary AITER/MoRI control;
- exact CUDA-graph replay stability within the upstream test contract;
- correct output slicing;
- valid routes at the highest expert ID;
- no deadlock under burst replay;
- all ranks make the same collective/protocol decision.

Required performance:

- use at least three repetitions after warmup;
- report rank mean and rank maximum latency;
- weight shape results by the AgentX manifest;
- enable MegaMoE only for buckets where it wins reproducibly;
- retain ordinary AITER fallback for unsupported or slower buckets.

Do not promote a kernel based only on one synthetic uniform-routing shape.

## 7. Phase 2B: Add a vLLM V4.1 adapter

### 7.1 Prototype location

The prototype belongs in a vLLM feature branch, not directly in InferenceX.
Likely touch points are:

- `vllm/models/deepseek_v41/amd/model.py`
- a new model-specific AITER MegaMoE adapter module;
- `vllm/_aiter_ops.py` or the modular MoE backend registry;
- ROCm MoE backend/config selection;
- model and kernel tests.

InferenceX should consume a pinned image built from that feature branch.

### 7.2 Adapter responsibilities

The adapter must:

1. Be explicitly opt-in and fail closed.
2. Require gfx950, A8W4, expert parallel, and a compatible AITER release.
3. Validate that experts divide the EP world size.
4. Validate top-k and V4.1 geometry.
5. Validate power-of-two MTPR and reachable token bounds.
6. Reject unsupported MoRI isolation mode.
7. Initialize MoRI symmetric memory on the EP CPU/process group.
8. Shuffle checkpoint expert weights/scales into A16W4 MegaMoEV2 layout after
   weight loading.
9. Cache MegaMoEV2 instances by rank, world size, layer shape, and MTPR.
10. Route through the existing V4.1 gate so text/image routing bias semantics
    are preserved.
11. Handle zero-token ranks with deterministic dummy routes so peers cannot
    deadlock.
12. Warm every reachable bundle/token bucket before graph capture.
13. Support both backbone and DSpark expert counts or explicitly fall back for
    draft layers.
14. Return the ordinary AITER path for unsupported or losing shapes.

The current V4.1 `use_mega_moe` and `finalize_mega_moe_weights()` placeholders
are the preferred model hooks.

### 7.3 Shared expert policy

For the first integration, run the native shared expert separately and overlap
it with routed MegaMoE only when stream and rank-synchronization semantics are
safe.

Do not fuse the shared expert into MegaMoEV2 in the first patch. That is a
separate optimization with different weight/scale and expert-count contracts.

### 7.4 vLLM test gates

Add CPU/mocked tests for:

- backend selection;
- fail-closed capability checks;
- unsupported hardware and quantization;
- missing AITER/MoRI;
- weight and scale conversion;
- backbone vs DSpark routing;
- empty-rank handling;
- MTPR validation;
- fallback behavior.

Add GPU tests for:

- real V4.1 routed checkpoint weights;
- EP4 and optional EP8;
- graph capture/replay;
- startup/shutdown;
- repeated decode;
- DSpark block and synthetic rejection modes;
- no leaked symmetric-memory state.

## 8. Phase 2C: Wire the candidate into InferenceX

After the vLLM and AITER gates pass:

1. Build and digest-pin a ROCm vLLM image containing the exact vLLM/AITER/MoRI
   revisions.
2. Add a separate MegaMoE recipe or explicit experimental arm. Do not silently
   change the ordinary AITER baseline.
3. Preserve TP/EP topology in result metadata.
4. Add all required environment/CLI propagation to the Ruby launcher.
5. Record MTPR, backend, AITER/MoRI revisions, and fallback counts.
6. Append only a new `perf-changelog.yaml` entry.
7. Update the nearest English and Chinese configuration documentation if the
   path becomes contributor-facing.

A large serving feature should not remain as an opaque runtime monkey patch.
During early bring-up, a checked-in, hash-verified overlay is acceptable for
attribution. The final candidate should use an immutable image from reviewed
vLLM/AITER source.

## 9. End-to-end benchmark matrix

### 9.1 Baseline/control/candidate arms

| Arm | Topology | Communication and MoE |
| --- | --- | --- |
| Original baseline | TP4 | Current ordinary AITER recipe |
| EP control | TP4/EP4 | MoRI + AITER CK A8W4 |
| Mega candidate | TP4/EP4 | AITER MegaMoEV2 |
| Optional EP8 control | TP8/EP8 | MoRI + AITER CK A8W4 |
| Optional EP8 Mega | TP8/EP8 | AITER MegaMoEV2 |

The attribution comparison is EP control versus Mega candidate at the same
topology. The original TP4 baseline remains necessary to determine whether
switching to expert parallel is worthwhile overall.

### 9.2 Run tiers

#### Tier 1: deterministic operator tests

Run the weighted AgentX shape manifest and routing distributions.

#### Tier 2: server synthetic tests

Use:

- decode-heavy small-M points;
- 8K/1K;
- long-prefill points;
- DSpark draft/verify;
- controlled route skew.

#### Tier 3: AgentX fast A/B

```text
concurrency 8
duration 1200 seconds
same node, image, model revision, trace seed, warmup, and telemetry
three serial repetitions per arm
```

Do not run control and candidate simultaneously on separate GPU halves;
concurrent graph capture and host contention can invalidate the comparison.

#### Tier 4: canonical AgentX

Run the selected candidate and its control for 3600 seconds at concurrency 8,
then widen to concurrency 1, 2, 4, 8, 16, and 32 if the candidate remains
promising.

#### Tier 5: accuracy

Run the full configured eval with real DSpark block rejection.

## 10. Acceptance criteria

### Correctness and stability

- Server reaches ready state on every rank.
- Zero model-load and scale-layout errors.
- Zero collective deadlocks or watchdog failures.
- Zero invalid expert IDs.
- Zero non-finite outputs.
- AgentX profiling request error rate is zero.
- Full eval score is at least the configured 0.90 floor.
- All requested graph buckets capture and replay.
- No unexplained fallback or mixed collective protocol across ranks.

### Performance

Predeclare P90 E2E normalized interactivity as the primary AgentX metric.
Also report:

- output throughput per GPU;
- TTFT;
- TPOT/ITL;
- E2E latency;
- request throughput;
- queue depth;
- GPU and host cache behavior;
- DSpark acceptance length/rate;
- power when available.

For fast A/B promotion:

- three serial matched runs per arm;
- median primary-metric improvement greater than 1%;
- output throughput is non-regressing within measurement noise;
- no material P95/P99 latency regression;
- the result agrees in direction with the weighted microbenchmark.

Confirm the selected candidate with at least one canonical 3600-second matched
run before making a performance claim.

## 11. Stop conditions

Stop and diagnose before widening if:

- the ordinary V4.1 baseline does not reproduce;
- exact image/model/AITER/MoRI provenance is unknown;
- the EP control cannot boot or pass accuracy;
- V4.1 checkpoint weights do not satisfy MegaMoEV2 layout assumptions;
- DSpark draft layers enter an unsupported path;
- any rank makes a different token-capacity or fallback decision;
- a smoke test hangs or emits invalid routes;
- a microbenchmark win disappears at server level;
- AgentX does not improve beyond noise;
- the candidate improves throughput by sacrificing correctness or tails.

Do not tune more token buckets merely to hide a failed end-to-end gate.

## 12. Deliverables

### InferenceX branch

- Ruby V4.1 preset.
- Any necessary generic launcher hardening.
- Exact config-generation and routing tests.
- Baseline and candidate recipe/config arms.
- Immutable image and source provenance.
- AgentX artifacts, summaries, and changelog entry.

### AITER branch

- V4.1 backbone and DSpark MegaMoEV2 test definitions.
- AgentX-derived shape/routing microbenchmark corpus.
- Any required V4.1 kernel configuration/tuning.
- Correctness, graph, skew, and performance evidence.

### vLLM branch

- Explicit AITER MegaMoEV2 adapter.
- Weight conversion and post-load hooks.
- EP/MoRI lifecycle.
- Backbone/draft fallback policy.
- Unit, distributed, and end-to-end tests.

### Optional Arbor follow-up

Once the deterministic shape corpus and vLLM integration exist, update Arbor
to optimize candidate AITER/Primus kernels against the microbenchmark and use
AgentX as the final KEEP/REVERT gate. Arbor automation is downstream of this
bring-up plan, not a prerequisite for the first working implementation.

## 13. Reference links

- DeepSeek V4.1 Flash model:
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- InferenceX MI355X bring-up:
  https://github.com/SemiAnalysisAI/InferenceX/pull/2962
- InferenceX CK A8W4 update:
  https://github.com/SemiAnalysisAI/InferenceX/pull/3058
- Validated InferenceX run:
  https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34710937012
- AITER MegaMoEV2:
  https://github.com/ROCm/aiter/pull/4439
- AITER v0.1.22:
  https://github.com/ROCm/aiter/releases/tag/v0.1.22
- SGLang MegaMoEV2 integration reference:
  https://github.com/sgl-project/sglang/pull/35619
- Alternative SGLang integration reference:
  https://github.com/sgl-project/sglang/pull/36269
- Historical removal of the old vLLM ROCm MegaMoE path:
  https://github.com/vllm-project/vllm/pull/43629
