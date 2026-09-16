# DeepSeek V4.1 Flash Selective vLLM MegaMoEV2 Adapter Plan

> Successor to the completed
> [pooled AITER workspace plan](./03_DSV41_FLASH_MEGAMOE_POOLED_AITER_WORKSPACE.md)
> under the
> [MegaMoE + AgentX master plan](./01_DSV41_FLASH_MEGAMOE_AGENTX.md).
> This is an internal execution document, not published InferenceX documentation.

Status: **smoke and bottleneck profile complete; stopped before 1200-second A/B**

## 0. Execution result (2026-09-16)

Implemented and validated:

- vLLM adapter commit
  `e01a3fe5ee727903c296379677ca31faf5d1accc`;
- AITER compatibility commit
  `22c82955b41e2b482a99290537a044e6964499b1`;
- one model-load-finalized workspace per EP rank: 39 symmetric allocations,
  1,695,265,648 bytes;
- rank-cached eager selection and unconditional graph, decode, DSpark, mixed,
  padded, unknown, and unlisted fallback, with DBO rejected at startup;
- source routing preservation for the current MoRI `combine()` API;
- collective workspace teardown before MoRI SHMEM finalization;
- immutable image
  `dsv41-megamoe-adapter:vllm-e01a3fe-aiter-22c8295`
  (`sha256:1d32c46232fcdd2924449f3dd60737cf008ba62c2a2438e7ee26875f8849695a`).

Reproducibility patch hashes:

- vLLM: `6984c3e13f5af046f60d7adf31275599d6859bf36c6d775125e14a0721429cee`;
- AITER: `c1f473b14ca6577fbf6553a77e44b211e24b1bf3b300e33d3cf465dc51948137`;
- MoRI: `fda470170f801d05ce68fcd19647e546c3c5f045f03e0244e22606af415c0b97`.

The first hybrid replay showed that AgentX chat adds one token relative to the
earlier eager capture. The exact allowlist was therefore extended only to the
observed pairs `888/889`, `2625/2626`, and `7100/7101`; TP4 operator tests
passed for all three +1 shapes.

The final hybrid candidate selected Mega exactly 40 times for each of M=889,
2626, and 7101 and zero times elsewhere. Every rank reported identical path
counters. Graph capture used ordinary Mori+AITER through the explicit
`stream_capture` fallback.

The matched 60-second directional smoke completed only two profiling requests
per arm, so it is not statistically useful. Candidate versus control was:

- request-latency average: +0.70%; P90: -0.14%;
- TTFT average: +3.96%; P90: +0.11%;
- ITL average: +1.48%; P90: +2.43%;
- request errors: zero in both arms.

This misses the >1% P90 improvement and <1% regression-risk gates. Per the stop
conditions, the three 1200-second repetitions and plan 05 were not started.
Durable evidence is under
`/home/jiaweche/dsv41-megamoe-validation-20260915/vllm-adapter/`.

### Bottleneck profile

The two-request P90s are not tail estimates: their absolute candidate-control
differences were −16.997 ms request latency, +1.074 ms TTFT, and +0.538 ms ITL.

A matched eager torch profile found:

- eligible request wall time improved 6.12% at M=889, 1.27% at M=2626, and
  5.39% at M=7101;
- total self CUDA time improved 6.91%;
- integrated `moe_forward_shared` CUDA time improved only 1.95%;
- decode-generation GPU time changed by only +0.27%;
- eager rank agreement costs 0.305 ms, but hybrid graph replay bypasses it;
- Mega covered only 120 of 6,440 observed non-capture layer calls (1.86%).

The dominant bottleneck is coverage. The trace had 5,960 ordinary unlisted
calls, primarily M≈16,376–16,380 chunks above MTPR=8192. Shared-expert overlap
and Mega Stage2/combine then hide most of the isolated operator gain on eligible
shapes. The next useful experiment is MTPR=16384 or an 8192 scheduler chunk cap,
not a longer serving A/B of the current selector.

## 1. Objective

Integrate the validated pooled AITER MegaMoEV2 workspace into pinned vLLM
without copying model weights and without changing decode, DSpark, mixed-batch,
or unsupported-topology behavior.

The adapter is experimental, opt-in, TP4/EP4-only, and fail-closed. Its first
goal is correctness and path attribution, not automatic backend selection.

## 2. Fixed prerequisites

Use:

```text
vLLM:
  /scratch/jiaweche/dsv41-megamoe/vllm-src
  base: eed1f3d0c6043bd494424a22443ee198dd56f657
  adapter: e01a3fe5ee727903c296379677ca31faf5d1accc

AITER:
  /scratch/jiaweche/dsv41-megamoe/aiter-vllm
  adapter: 22c82955b41e2b482a99290537a044e6964499b1

MoRI library SHA256:
  3df6da1342f1c9dc7923fd2620bb132b283b2063bf0040a88cb08056e136cfd5

Model:
  deepseek-ai/DeepSeek-V4.1-Flash
  dba1be0a40aa45a94ad051997016db3960a90277
```

The pooled workspace is fixed at MTPR 8192 for the first adapter and measures
approximately 2.24 GB decimal per TP4 rank.

## 3. Integration seam

Keep the existing vLLM routing and shared-expert orchestration.

Branch in:

```text
vllm/model_executor/layers/quantization/mxfp4.py
Mxfp4MoEMethod.apply
```

At that point vLLM has already produced:

```text
hidden states
top-k weights
top-k IDs
finalized local w13/w2 weights and scales
shared-expert wrapper and input
```

The method should:

1. ask the selector for one cached decision for the current model forward;
2. call the pooled `MegaMoEV2.forward_with_weights` when eligible;
3. otherwise invoke the existing modular Mori plus ordinary-AITER kernel
   unchanged.

Both candidate and fallback routed outputs are already reduced. Preserve the
existing `MoERunner` shared-expert overlap, separate shared-output reduction,
and final add.

Do not add a model-level external router or duplicate `FusedMoEFactory`.

## 4. Configuration

Add an orthogonal experimental kernel configuration rather than replacing the
ordinary AITER backend required for fallback.

The configuration must carry:

```text
enabled: false by default
candidate MTPR: 8192
exact allowed raw M values: 888/889, 2625/2626, 7100/7101
required global experts: 384
required top-k: 6
required model/intermediate dimensions: 5120/2304
```

Startup must reject an enabled adapter when:

- the platform is not gfx950;
- expert parallel is disabled;
- EP world size is not four;
- DP, PP, PCP, or sequence parallel exceeds one;
- dual-batch overlap is enabled;
- the all-to-all backend is not intranode MoRI;
- EPLB or redundant experts are enabled;
- the MXFP4 backend does not produce AITER A16W4 shuffled weights;
- a local weight is non-contiguous, unmarked, wrong-shaped, or wrong-dtyped;
- scheduler capacity cannot cover MTPR 8192.

An unsupported configuration is a startup error when the explicit adapter flag
is enabled. A runtime shape/phase mismatch is an ordinary fallback.

## 5. Rank-consistent runtime selector

Use `ForwardContext`, not `hidden_states.shape[0]` alone.

The first MoE layer in each model forward computes and caches the decision in
`ForwardContext.additional_kwargs`. Every later backbone layer reuses it.

Candidate eligibility requires:

- forward context is available;
- CUDA/HIP graph runtime mode is `NONE`;
- no dual-batch-overlap microbatch slices;
- DeepSeek sparse-SWA metadata is available;
- zero decode rows and at least one prefill row;
- no mixed or extend phase;
- no padding rows;
- unpadded raw M is exactly 888/889, 2625/2626, or 7100/7101;
- the model is the 384-expert, top-k-6 backbone, not DSpark;
- every EP rank agrees on phase, M, and eligibility.

Perform one min/max agreement across the EP group when creating the cached
decision. Do not add a collective per layer.

Fallback unconditionally for:

- backbone decode M=6;
- DSpark M=5;
- the prior uniform M=16 regression;
- mixed, extend, padded, DBO, or unknown phases;
- FULL or PIECEWISE graph replay;
- any M not in the exact allowlist;
- any rank disagreement.

The current recipe captures graphs only through M=128, so the three allowed
prefill values should remain eager while decode retains its existing graph.

## 6. Workspace ownership

Let `MoriAll2AllManager` own exactly one pooled MegaMoEV2 workspace for its EP
group.

The manager must:

- initialize the workspace only after MoRI SHMEM and finalized model weights
  are available on every rank;
- return the same workspace to every eligible backbone MXFP4 method;
- never create a workspace for DSpark;
- expose allocation count and bytes in startup receipts;
- close the workspace before destroying MoRI handles or process groups;
- call close collectively and idempotently;
- clear borrowed weight references only after device synchronization.

Do not create one adapter module or workspace per model layer.

## 7. Unit and distributed tests

Add behavioral tests for:

- static capability validation;
- pure-prefill allowlist selection;
- decode, DSpark, mixed, extend, padding, graph, DBO, and unknown fallback;
- one cached EP agreement per model forward;
- simulated rank disagreement;
- exact reuse of existing weight storage pointers;
- strict rejection of unshuffled explicit-path weights;
- existing shared-expert output parity and reduction;
- one workspace returned across 40 backbone layers;
- manager destroy calls workspace close before communicator destruction;
- double shutdown and model reload in one process.

On four gfx950 GPUs, compare adapter output against the ordinary EP control for:

- captured M=888;
- captured M=2625;
- captured and count-exact hot-skew M=7100;
- decode M=6 fallback;
- DSpark M=5 fallback;
- mixed/unknown fallback injection.

Require no invalid routes, non-finite output, deadlock, or residual VRAM.

## 8. Image and server gates

Build one immutable image from exact vLLM, AITER, and MoRI revisions.

Run a TP4/EP4 smoke with:

```text
--enable-expert-parallel
--all2all-backend mori_high_throughput
ordinary AITER MXFP4 backend retained
selective MegaMoEV2 adapter enabled
max graph capture size 128
```

Emit counters by component, phase, raw M, selected path, and fallback reason.

The first 60-second replay must prove:

- Mega calls occur only for backbone M=888/889, 2625/2626, and 7100/7101
  pure prefill;
- zero Mega calls occur for backbone decode and DSpark;
- all ranks report identical decisions and call counts;
- exactly one workspace is allocated per rank;
- memory delta agrees with the pooled operator result;
- shutdown exits cleanly with zero residual containers and VRAM;
- matched control/candidate outputs and accuracy remain inside existing gates.

## 9. Fast matched A/B

Only after the smoke and path counters pass, run three serial 1200-second
AgentX concurrency-8 repetitions per arm:

```text
control:
  TP4/EP4 MoRI + ordinary AITER

candidate:
  same topology and image, selective prefill MegaMoEV2 enabled
```

Hold model revision, trace, seed, warmup, graph policy, scheduler, telemetry,
and power collection fixed.

Advance only when:

- median P90 normalized interactivity improves by more than 1%;
- output throughput does not regress beyond noise;
- TTFT improves in the expected direction;
- decode ITL and tails do not regress materially;
- request error rate is zero;
- measured path counts explain the result.

## 10. Artifacts

Store durable evidence under:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/vllm-adapter/
```

Persist revisions, patches, image identity, startup receipts, selector
counters, memory JSON, server logs, request outputs, accuracy results, three-run
A/B summaries, and a final `RESULTS.md`.

Commit plans and small reproducibility overlays to the InferenceX branch.
Keep checkpoints, image layers, build products, and compiler caches on
`/scratch`.

## 11. Stop conditions

Stop before AgentX if:

- weight storage is copied or replaced;
- more than one workspace appears per rank;
- shared-expert parity fails;
- a rank selects a different path;
- candidate code runs under decode or DSpark;
- a graph captures a dynamic selector branch;
- workspace close occurs after communicator teardown;
- any smoke exits with residual VRAM or a core dump;
- MTPR or routing differs between control and candidate.

## 12. Next step after this plan

After the three matched fast A/B repetitions pass, create:

```text
plans/05_DSV41_FLASH_MEGAMOE_AGENTX_CONFIRMATION.md
```

That plan owns the canonical 3600-second matched confirmation, concurrency
widening, real DSpark block-rejection accuracy, final KEEP/REVERT decision, and
performance changelog.
