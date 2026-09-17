# DeepSeek V4.1 Flash Async MegaMoE Selection Plan

> Successor to the stopped
> [selector remediation](./07_DSV41_FLASH_MEGAMOE_SELECTOR_REMEDIATION.md)
> under the
> [MegaMoE + AgentX master plan](./01_DSV41_FLASH_MEGAMOE_AGENTX.md).
> This is an internal execution document, not published InferenceX documentation.

Status: **finite-corpus gates passed; looped canonical pending**

## 0. Execution progress (2026-09-17)

The first implementation:

- launches a context-owned strict Gloo gather asynchronously;
- drains unconsumed work on context exit and exceptions;
- skips runtime tensor validation on metadata-ineligible fallback;
- retains eligible-only runtime agreement;
- preserves 40-layer path-counter totals with context-level accounting.

Focused source and image suites pass 43/43. Immutable image:

```text
vLLM:
  acb139b8725a46e1747c163f3d57089ff8bffee4

image:
dsv41-megamoe-adapter:vllm-e2944d9-aiter-22c8295-async-r1
sha256:3c8ed972808c0530383281c71a337ca62471c70608d7a3fbb55adf4f9865855b
```

The matched trace gate passes:

- Gloo completes 17–42 ms before first-MoE selection;
- first-MoE completion checks take 46–49 us;
- fallback adapter time falls from 1.578 to 0.380 ms/context;
- exact mixed M=16,376 Mori subtree: 0.26% faster;
- exact mixed M=16,376 Mori combine: 4.34% faster;
- mixed-context mean wall time: 0.47% faster;
- zero Mega kernels in the fallback window;
- four identical rank receipts and zero rank disagreements;
- healthy post-export server and zero residual VRAM.

Profiler step scheduling produced three control and two candidate mixed
iterations. The gate therefore compares exact-shape per-layer means and
mixed-context distributions rather than aggregate trace totals.

Three 1200-second pairs completed 237 profiled requests per arm with zero
errors. Pairwise candidate changes:

- pair 1: interactivity +7.35%, TTFT −6.83%, ITL P90 +2.85%,
  ITL P99 +1.83%;
- pair 2: interactivity +0.84%, TTFT +1.94%, ITL P90 −0.43%,
  ITL P99 −0.59%;
- pair 3: interactivity +4.23%, TTFT +4.26%, ITL P90 +0.39%,
  ITL P99 −1.78%.

Pair 1's TTFT failure was not systematic: its matched-request median improved
0.73 ms, while >100 ms shifts were nearly balanced at 22 regressions and 19
improvements. The two adjudication pairs were therefore run under the explicit
rule in section 6.

Median candidate changes:

- P90 normalized interactivity: **+4.23%**;
- TTFT P90: **+1.94%**;
- E2EL P90: **−0.14%**;
- ITL P90: **+0.39%**;
- ITL P99: **−0.59%**;
- output throughput: **−0.01%**;
- active GPU 0–3 power: **+0.004%**.

All median gates pass and two of three pairs pass the primary interactivity
gate. Median Mega coverage is 37.25%; all 12 rank receipts are identical and
contain zero rank disagreements.

Full 1,319-example GSM8K with real DSpark block rejection reproduces the
passing candidate score: 0.9727 flexible / 0.9735 strict exact match.

Decision: keep the async implementation and image as the next candidate. Do
not issue final canonical KEEP or rerun nominal 3600 seconds until the finite
AgentX corpus is explicitly looped; the current runner adds idle time rather
than samples.

## 1. Objective

Remove synchronous candidate-only selection work from the first MoE of eager
mixed fallback iterations while preserving fail-closed rank agreement.

The Plan 07 follow-up profile isolates the residual canonical ITL P90 miss:

- one steady Gloo gather takes 0.318 ms / 0.672 ms inclusive Python;
- cached adapter bookkeeping costs approximately 0.424 ms per 40 layers;
- first mixed Mori dispatch starts 0.378 ms later;
- mixed-context wall time rises 1.20%;
- M=16,372 Mori combine rises 10.50%;
- expert stage 1+2 kernels are 7.43% faster;
- neither matched trace executes Mega or shows a decode-kernel regression.

## 2. Fixed baseline

Keep fixed:

```text
vLLM base:
  e2944d9ceab9219d7f41dcb621322d91d7967e66

Plan 07 remediation:
  132d417f020c37d02a101bf06b410d4db1a27ae3

AITER:
  22c82955b41e2b482a99290537a044e6964499b1

topology:
  TP4 / EP4 / DP1 / PP1 / PCP1

MTPR:
  16384

graph policy:
  compilation mode NONE
  FULL_DECODE_ONLY
```

The Mega kernel, workspace, weights, allowlist, Mori backend, model, corpus,
DSpark configuration, and serving limits must not change.

## 3. Design

### 3.1 Start shape agreement with `ForwardContext`

When `set_forward_context` creates a Mega-enabled eager context, compute the
metadata-only local decision and launch one strict CPU Gloo
`all_gather_into_tensor(..., async_op=True)`.

Store a context-owned agreement object containing:

- immutable local decision and raw M;
- dedicated local and gathered CPU tensors;
- the distributed `Work` handle;
- an idempotent completion result.

Do not reuse buffers while work is in flight.

Graph capture remains fail-closed and launches no collective.

### 3.2 Consume safely at first MoE

The first adapter invocation waits for the context agreement only if it has not
already completed. All 40 layers reuse the completed decision.

Any eligibility or raw-M disagreement returns ordinary Mori with
`rank_disagreement`.

The context manager drains unconsumed work on normal exit and exceptions so no
collective or buffer survives the context.

### 3.3 Preserve runtime-contract agreement

Runtime tensor checks are unnecessary for a metadata-ineligible fallback.

For a globally shape-eligible batch:

1. validate runtime tensors locally at the first MoE;
2. perform strict runtime agreement before entering Mega;
3. fail closed if any rank rejects the runtime contract.

An extra eligible-only agreement is acceptable initially; it must not appear
on mixed, decode, graph, or unlisted paths.

### 3.4 Collapse fallback accounting

Record routing counters once per `ForwardContext` while preserving existing
layer-call totals by applying the fixed 40-layer multiplier.

No performance run may lose path attribution.

## 4. Unit gates

Tests must prove:

- async agreement starts exactly once per eligible configuration context;
- the first MoE consumes it and later layers issue no collective;
- completion is idempotent;
- exception and unused-context exits drain work;
- buffers are not reused across in-flight contexts;
- graph capture starts no collective;
- fallback skips runtime tensor validation;
- eligible runtime disagreement fails closed;
- shape or eligibility disagreement fails closed;
- counter totals remain backward-compatible;
- all existing adapter and lifecycle tests pass.

## 5. Image and live trace gates

Build a new immutable image only after formatting, lint, focused tests, and
patch reproducibility pass.

Repeat the exact matched Plan 07 P90 trace:

- pure unlisted M=5,206;
- mixed M=16,360 plus 12 decode rows;
- routed ordinary-Mori M=16,372;
- two profiler-active model iterations.

Require:

- no synchronous selector collective immediately before ordinary Mori;
- mixed first-dispatch delta within ±0.2 ms;
- mixed-context wall delta within ±0.2%;
- no Mega kernel in the fallback window;
- identical four-rank counters and zero disagreements;
- healthy post-export server and zero residual VRAM.

Stop before A/B if any trace gate fails.

## 6. Performance gates

If the trace passes, run one matched 1200-second C8 control/candidate pair.
Require:

- P90 normalized interactivity improvement greater than 1%;
- TTFT P90 improvement greater than 1%;
- ITL P90 and P99 no worse than +1%;
- E2EL P90 no worse than +1%;
- output throughput no worse than −0.5%;
- zero request errors and zero residual state.

If it passes, run two additional pairs and apply the existing median and
two-of-three gates.

If only TTFT P90 fails while interactivity and both ITL tails pass, inspect
matched-request TTFT first. Two adjudication pairs may proceed when the matched
median is within ±1 ms and >100 ms shifts are bidirectional; the final
three-pair median TTFT gate remains unchanged.

Do not repeat the nominal 3600-second command until the finite AgentX corpus is
explicitly looped; the prior command added idle wall time but no samples.

## 7. Artifacts

Store evidence under:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/mtpr16384/async-selection/
```

Persist design, code diff, tests, image receipt, all-rank traces, TraceLens
reports, paired metrics, counters, cleanup receipts, and the final decision.
