# DeepSeek V4.1 Flash MTPR=16384 AgentX Confirmation Plan

> Successor to the directionally passing
> [MTPR=16384 plan](./05_DSV41_FLASH_MEGAMOE_MTPR16384.md)
> under the
> [MegaMoE + AgentX master plan](./01_DSV41_FLASH_MEGAMOE_AGENTX.md).
> This is an internal execution document, not published InferenceX documentation.

Status: **stopped at fast-pair ITL P99 gate**

## 0. Execution result (2026-09-16)

Completed three serial 1200-second control/candidate pairs. Every arm completed
237 profiling requests with zero request errors, clean shutdown, and zero
residual VRAM.

Median candidate improvement:

- P90 normalized interactivity: +2.63% — pass;
- TTFT P90: +2.55% — pass;
- E2EL P90: +0.13% — pass;
- ITL P90: +0.85% — pass;
- output throughput: −0.09% — pass;
- active-GPU mean power: −0.29%;
- ITL P99: **−2.21% — fail**.

Two of three pairs passed the primary normalized-interactivity gate. Pairwise
ITL P99 candidate changes were −2.21%, −15.84%, and +4.77%, so the median
exceeds the permitted +1% regression.

Per-request matching found only +0.049/+0.015/+0.007 ms median ITL changes
across the three pairs, but a small set of requests regressed by 5–34 ms and
drives the failed P99. The tail risk is not a broad decode slowdown, but it is
large enough that the gate cannot be waived.

All ranks reported identical path counters and one workspace:

- pair 1: 5,960 Mega calls, 37.25% non-capture coverage;
- pair 2: 5,840 Mega calls, 36.23% coverage;
- pair 3: 5,920 Mega calls, 36.91% coverage.

Long-run coverage is lower than the directional smoke because approximately
9,600 mixed-phase layer calls per run correctly remain on ordinary Mori+AITER.

Power collection observed all eight physical GPUs instead of the four visible
model GPUs, so built-in validation reports `expected_gpu_count_mismatch`.
Filtered GPU 0–3 estimates show 0.18–0.64% lower candidate power.

Per the stop gate, block-rejection accuracy, the 3600-second pair, and
concurrency widening were not started. The candidate is not approved for KEEP.
Durable evidence is under
`/home/jiaweche/dsv41-megamoe-validation-20260915/mtpr16384/confirmation/`.

## 1. Objective

Determine whether the MTPR=16384 selective MegaMoEV2 adapter produces a
repeatable AgentX serving improvement after the two-request directional smoke.

This plan owns statistically useful confirmation. The plan 05 smoke is not
enough for a KEEP decision.

## 2. Fixed candidate

```text
vLLM:
  e2944d9ceab9219d7f41dcb621322d91d7967e66

AITER:
  22c82955b41e2b482a99290537a044e6964499b1

MTPR:
  16384

allowlist:
  888, 889, 2625, 2626, 7100, 7101, 16376, 16380

graph policy:
  compilation mode NONE
  FULL_DECODE_ONLY
  max capture size 128
```

Use the same immutable image for both arms. The control enables
`force_mori_all2all`; the candidate enables `enable_aiter_mega_moe_v2`.

## 3. Fast confirmation

Run three serial 1200-second repetitions per arm at AgentX concurrency 8:

```text
control, candidate, control, candidate, control, candidate
```

Hold fixed:

- model and tokenizer revisions;
- trace corpus, slice, seed, and lane start positions;
- TP4/EP4/DP1/PP1/PCP1 topology;
- DSpark synthetic acceptance configuration;
- scheduler capacity 16384;
- graph policy and capture sizes;
- GPU-memory utilization 0.8;
- warmup requests and Mega startup preload;
- telemetry and power collection.

Do not run arms concurrently. Verify zero residual VRAM between runs.

## 4. Required evidence

For every run persist:

- full AIPerf exports and request accounting;
- vLLM server log and command;
- per-rank Mega path counters;
- workspace startup receipt;
- GPU memory and power;
- request error breakdown;
- normalized interactivity, TTFT, E2EL, ITL, throughput, and cache metrics.

Require counters to show:

- identical values across ranks;
- Mega at M=16,376 and M=16,380;
- ordinary fallback for decode, DSpark, graph capture, mixed, padded, and
  unlisted shapes;
- exactly one workspace per rank.

## 5. Advance gates

Advance when the median of three candidate-control pairs shows:

- P90 normalized interactivity improvement greater than 1%;
- TTFT P90 improvement greater than 1%;
- output throughput no worse than −0.5%;
- ITL P90 and P99 no worse than +1%;
- E2EL P90 no worse than +1%;
- zero request errors;
- no residual process, VRAM, or core dump.

Treat one regressing pair as noise only if the median and two of three pairs
pass. Otherwise stop and profile that pair.

## 6. Accuracy

Run real DSpark block-rejection accuracy on the fixed candidate and control.
Require existing model/evaluation gates and no new non-finite outputs.

## 7. Canonical confirmation

If the three fast pairs pass, run one matched 3600-second pair at concurrency 8,
then widen to the existing supported concurrency range.

Produce a final KEEP/REVERT decision with measured path attribution.

## 8. Artifacts

Store durable evidence under:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/mtpr16384/confirmation/
```
