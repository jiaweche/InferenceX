# DeepSeek V4.1 Flash MegaMoE Selector Remediation Plan

> Successor to the stopped
> [MTPR=16384 confirmation](./06_DSV41_FLASH_MEGAMOE_AGENTX_CONFIRMATION.md)
> under the
> [MegaMoE + AgentX master plan](./01_DSV41_FLASH_MEGAMOE_AGENTX.md).
> This is an internal execution document, not published InferenceX documentation.

Status: **stopped at canonical ITL P90 gate**

## 0. Execution result (2026-09-16)

Implementation and trace gates pass.

- metadata-first fallback avoids the selector padding tensor for mixed, decode,
  and unlisted batches;
- strict agreement uses one preallocated Gloo gather instead of two reductions;
- focused source and image suites: 33/33 pass;
- immutable image:
  `dsv41-megamoe-adapter:vllm-e2944d9-aiter-22c8295-selector-r2`;
- image ID:
  `sha256:22ef7b63b4a671c8b2374b309e91ea4a93e484ae11d642e801d4c2209c5b5a14`.

The first image revision exposed a Gloo-specific output-layout constraint during
startup preflight and served no requests. Revision two uses the required flat
concatenated gather buffer and passed startup.

The revision-two Chrome trace captured one pure-unlisted and one mixed model
iteration:

- selector-owned `aten::any` events: zero;
- Gloo selection agreements: two, exactly one per model iteration;
- mean Gloo agreement: 0.353 ms;
- Mega kernels: zero, as required for the fallback-only window;
- four-rank path counters: identical, with zero rank disagreements;
- server healthy after trace export and zero residual VRAM after shutdown.

Durable implementation and trace receipts are under
`/home/jiaweche/dsv41-megamoe-validation-20260915/mtpr16384/remediation/`.

Three 1200-second pairs completed 237 profiled requests per arm with zero
errors. Median candidate changes:

- P90 normalized interactivity: +6.13%;
- TTFT P90: +2.24%;
- E2EL P90: −0.38%;
- ITL P90: +1.16%;
- ITL P99: +5.35%;
- output throughput: +0.04%;
- active GPU 0–3 power: −0.33%.

All fast-pair median gates pass, including two of three primary interactivity
passes. Median candidate routing coverage is 37.37%; all rank receipts are
identical and contain zero rank disagreements.

Full 1,319-example GSM8K with real DSpark block rejection also passes:
candidate exact match is 0.9727 flexible / 0.9735 strict versus control
0.9704 / 0.9712.

The canonical pair retains the primary gain but narrowly fails the strict ITL
P90 gate:

- P90 normalized interactivity: +3.59%;
- TTFT P90: +6.51%;
- E2EL P90: −0.03%;
- ITL P90: **−1.03%**;
- ITL P99: −0.08%;
- output throughput: −0.06%;
- active GPU 0–3 power: −0.21%.

The original ITL P99 regression is removed, but ITL P90 exceeds the allowed
regression by approximately 0.03 percentage points. Per the stop gate,
concurrency widening was not started and the candidate is not approved for
canonical KEEP.

The AgentX corpus is finite: the nominal 3600-second pair produced the same 237
profiled requests and approximately 1,226 seconds of active workload as each
fast arm. The remaining wall time added no samples, so a future long
confirmation must explicitly repeat the corpus.

## 1. Objective

Remove candidate-only synchronization from mixed and unlisted eager fallback
batches without weakening fail-closed rank agreement or changing the validated
Mega allowlist, workspace, graph policy, or operator.

The remediation targets the TraceLens findings from Plan 06:

- zero Mega kernels in the captured ITL-tail window;
- one GPU `aten::any().item()` padding check per model iteration;
- two Gloo all-reduces per model iteration;
- 0.579 ms measured Gloo time per mixed model iteration.

## 2. Fixed baseline

Keep fixed:

```text
vLLM:
  e2944d9ceab9219d7f41dcb621322d91d7967e66

AITER:
  22c82955b41e2b482a99290537a044e6964499b1

MTPR:
  16384

allowlist:
  888, 889, 2625, 2626, 7100, 7101, 16376, 16380

topology:
  TP4 / EP4 / DP1 / PP1 / PCP1

graph policy:
  compilation mode NONE
  FULL_DECODE_ONLY
  max capture size 128
```

Only the selector and its CPU agreement primitive may change.

## 3. Implementation

### 3.1 Metadata-first local selection

Evaluate graph mode, DBO state, sparse-attention phase metadata, pure-prefill
shape, and allowlist membership before reading `ForwardContext.is_padding`.

Only a locally eligible pure-prefill allowlisted batch may execute the GPU
padding reduction. Mixed, decode, graph, DBO, missing/disagreeing metadata,
not-pure-prefill, and unlisted batches must not launch `aten::any`.

### 3.2 One strict rank agreement

Replace the MIN and MAX Gloo all-reduces with one
`all_gather_into_tensor` over `(eligible, raw_m)`.

Requirements:

- use preallocated CPU tensors owned by `MoriAll2AllManager`;
- serialize access with the existing manager lock;
- require every gathered row to equal the local row;
- return ineligible on any rank disagreement;
- retain one agreement per `ForwardContext`, shared by all 40 layers;
- reuse the same primitive for startup layer preflight.

No hash, sum, or average encoding is allowed because it could produce
rank-asymmetric agreement under adversarial values.

## 4. Unit gates

Add tests proving:

- mixed, decode, and unlisted decisions never touch the padding tensor;
- an otherwise eligible batch still rejects real padding;
- exactly one distributed collective is issued per uncached decision;
- identical gathered decisions preserve eligibility;
- any eligibility or M disagreement fails closed;
- cached decisions still issue no additional collective;
- all existing adapter, lifecycle, graph, and ordinary-Mori tests pass.

Run the focused adapter suite and formatter/linter checks.

## 5. Image and trace gates

Build a new immutable adapter image and record its ID and source diff.

Collect a short candidate Chrome trace over the same tail workload. Require:

- zero Mega kernels in mixed fallback iterations;
- zero selector-owned `aten::any` events in mixed/unlisted iterations;
- one Gloo agreement event per uncached model iteration, not two;
- identical decisions and path counters on all four ranks;
- no startup, request, shutdown, or residual-VRAM failure.

If the trace gate fails, stop before performance confirmation.

## 6. Fast confirmation

Using the new image for both arms, run three serial 1200-second pairs:

```text
control, candidate, control, candidate, control, candidate
```

Use the exact Plan 06 corpus, lane starts, seed, concurrency 8, DSpark
configuration, telemetry, and accounting.

Advance only when the median of three candidate-control pairs shows:

- P90 normalized interactivity improvement greater than 1%;
- TTFT P90 improvement greater than 1%;
- output throughput no worse than −0.5%;
- ITL P90 and P99 no worse than +1%;
- E2EL P90 no worse than +1%;
- zero request errors;
- no residual process, VRAM, or core dump.

Two of three pairs must pass the primary interactivity gate. Treat a single
tail regression as noise only when the median and two of three pairs pass.

## 7. Conditional completion

If fast confirmation passes:

1. run real DSpark block-rejection accuracy;
2. run one matched 3600-second pair at concurrency 8;
3. widen across the existing supported concurrency range;
4. issue a final KEEP or REVERT decision.

Stop at the first failed gate and preserve all evidence.

## 8. Artifacts

Store durable evidence under:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/mtpr16384/remediation/
```

Required artifacts include the code diff, unit-test receipt, image receipt,
trace receipt, per-arm benchmark exports, path counters, telemetry, paired
summary, and final decision.
