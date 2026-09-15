# DeepSeek V4.1 Flash Selective vLLM MegaMoEV2 Adapter Plan

> Successor to the completed
> [pooled AITER workspace plan](./03_DSV41_FLASH_MEGAMOE_POOLED_AITER_WORKSPACE.md)
> under the
> [MegaMoE + AgentX master plan](./01_DSV41_FLASH_MEGAMOE_AGENTX.md).
> This is an internal execution document, not published InferenceX documentation.

Status: **next**

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
  eed1f3d0c6043bd494424a22443ee198dd56f657

AITER:
  /scratch/jiaweche/dsv41-megamoe/aiter-vllm
  e4b600af4ed97453a325711e2fc2aee0ebeb5239

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
exact allowed raw M values: 888, 2625, 7100
required global experts: 384
required top-k: 6
required model/intermediate dimensions: 5120/2304
```

Startup must reject an enabled adapter when:

- the platform is not gfx950;
- expert parallel is disabled;
- EP world size is not four;
- DP, PP, PCP, or sequence parallel exceeds one;
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
- unpadded raw M is exactly 888, 2625, or 7100;
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

- Mega calls occur only for backbone M=888, 2625, and 7100 pure prefill;
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
