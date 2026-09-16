# DeepSeek V4.1 Flash MegaMoEV2 MTPR=16384 Plan

> Successor to the profiled
> [selective vLLM adapter plan](./04_DSV41_FLASH_MEGAMOE_VLLM_ADAPTER.md)
> under the
> [MegaMoE + AgentX master plan](./01_DSV41_FLASH_MEGAMOE_AGENTX.md).
> This is an internal execution document, not published InferenceX documentation.

Status: **directional serving gate passed**

## 0. Execution result (2026-09-16)

Operator gates passed:

- M=16,376 median speedup: +43.01% across three repetitions;
- M=16,380 median speedup: +42.71% across three repetitions;
- M=16,380 reference accuracy: relL2 0.058276/0.058270;
- graph replay, A/B/A weight rebinding, close/recreate, and zero steady growth
  passed;
- one workspace per rank uses 39 symmetric allocations, 3,324,097,392
  symmetric bytes, and 1,052,546,560 CUDA workspace bytes.

The committed vLLM profile is
`e2944d9ceab9219d7f41dcb621322d91d7967e66`. The immutable image is
`dsv41-megamoe-adapter:vllm-e2944d9-aiter-22c8295`
(`sha256:4d659a1f1312b6e79fc3a2df9801dd7b11337784a37b629b591e51cd61587c40`).
The vLLM patch SHA256 is
`10df3a9443805638772c8ed680e75e37be9467710771b37936a24fe3877ae191`.

The default vLLM compilation mode performed a padded full-model dummy run at
exactly M=16,384 and reproducibly faulted inside sparse attention. Lowering GPU
memory utilization and disabling registered JIT warmups did not change the
fault. The supported serving policy is compilation mode `NONE` with
`FULL_DECODE_ONLY`: prefill remains eager, decode retains graphs, and the unsafe
full-prefill compile pass is absent.

Under that policy, TP4/EP4 DSpark serving passed:

- M=16,380 and M=16,376 selected Mega for all 40 backbone layers;
- neighboring M=16,379 and decode used ordinary fallback;
- graph capture and DSpark remained ordinary;
- every rank reported identical counters;
- one workspace per rank and 129.96 GiB available KV cache;
- all requests succeeded and shutdown left zero residual VRAM.

The matched C8 60-second directional smoke completed two profiled requests per
arm. Candidate versus control:

- P90 normalized interactivity: +3.20%;
- TTFT P90: 4.33% faster;
- E2EL P90: 0.44% faster;
- ITL P90/P95: 1.59%/1.65% faster;
- output throughput: +0.75%;
- request errors: zero;
- observed non-capture Mega coverage: 91.93% (5,920 of 6,440 layer calls).

The directional gate passes, but n=2 is not confirmation. Plan 06 owns the
three 1200-second pairs and canonical confirmation. Durable evidence is under
`/home/jiaweche/dsv41-megamoe-validation-20260915/mtpr16384/`.

## 1. Objective

Raise the pooled AITER MegaMoEV2 workspace from MTPR=8192 to MTPR=16384 so
the dominant AgentX prefill chunks at raw M=16,376 and M=16,380 become eligible.
Measure whether the coverage increase survives integrated shared-expert overlap
and clears the serving gate without changing graph decode or DSpark behavior.

The MTPR=8192 result remains a completed, stopped experiment. This plan creates
a new candidate; it does not reinterpret the prior two-request A/B as a pass.

## 2. Fixed baseline

Use:

```text
vLLM adapter:
  e01a3fe5ee727903c296379677ca31faf5d1accc

AITER adapter:
  22c82955b41e2b482a99290537a044e6964499b1

MoRI library SHA256:
  3df6da1342f1c9dc7923fd2620bb132b283b2063bf0040a88cb08056e136cfd5

Model:
  deepseek-ai/DeepSeek-V4.1-Flash
  dba1be0a40aa45a94ad051997016db3960a90277
```

Preserve TP4/EP4, intranode Mori, ordinary AITER fallback, PP1/DP1/PCP1,
sequence-parallel off, DBO off, EPLB off, and graph capture ceiling 128.

## 3. Hypothesis

The MTPR=8192 hybrid trace selected Mega for only 120 of 6,440 observed
non-capture backbone layer calls (1.86%). It left 5,960 unlisted calls,
primarily:

```text
M=16,380: 5,400 layer calls
M=16,376:   520 layer calls
```

Adding those two exact shapes would raise observed non-capture coverage to
approximately 93.8%. This can produce an end-to-end gain only if Mega remains
faster than ordinary Mori+AITER at M≈16K and the larger pool does not create
memory, clock, or teardown regressions.

## 4. Operator gates

Create the workspace with:

```text
max_tok_per_rank=16384
```

On four gfx950 GPUs require:

- construction, one forward, alternating explicit weights, graph replay, close,
  and close/recreate at M=16,376 and M=16,380;
- finite output and no invalid routes;
- relL2 within the existing FP8xFP4 gate where a reference is practical;
- exactly one workspace per rank;
- measured symmetric allocation count and bytes;
- no residual VRAM or process after close;
- three serial ordinary-Mori versus Mega repetitions at both dominant shapes.

Advance to vLLM only if median Mega routed-MoE time improves by more than 5% at
both shapes. Stop immediately if either shape regresses.

## 5. Adapter changes

Add an MTPR=16384 rollout profile:

```text
candidate MTPR: 16384
exact raw-M allowlist:
  888, 889
  2625, 2626
  7100, 7101
  16376, 16380
```

Keep the MTPR in the workspace key and startup receipt. Require scheduler
capacity of at least 16384. Preserve all existing rank agreement, model-load
weight preflight, graph fallback, one-workspace ownership, and teardown rules.

Do not route arbitrary M values below 16384; only the measured shapes are
eligible.

## 6. Test and image gates

Update unit tests for:

- the fixed MTPR=16384 profile and expanded exact allowlist;
- rejection when scheduler capacity is below 16384;
- selection at M=16,376 and M=16,380;
- fallback at neighboring unmeasured values;
- unchanged decode, DSpark, mixed, padded, graph, and DBO behavior;
- model reload and one pooled workspace across 40 layers.

Build one immutable image from committed vLLM, AITER, and MoRI revisions.
Record its ID and rerun the combined ordinary/Mega image smoke.

## 7. Full-model and AgentX gates

Run a TP4/EP4 DSpark smoke with exact M=16,376 and M=16,380 requests. Require:

- 40 Mega calls per exact request on every rank;
- zero Mega calls for decode and DSpark;
- identical rank counters;
- one workspace per rank;
- successful output and clean shutdown;
- enough remaining KV capacity for the fixed AgentX recipe.

Then run matched hybrid C8 control/candidate smokes with identical trace, seed,
warmup, graph policy, scheduler, power collection, and image.

Advance to long A/B only when:

- candidate P90 normalized interactivity improves by more than 1%;
- TTFT improves in the expected direction;
- output throughput does not regress beyond noise;
- ITL P90/P99 remains within 1%;
- request errors are zero;
- counters prove M≈16K coverage.

## 8. Artifacts

Store durable evidence under:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/mtpr16384/
```

Persist operator JSON/logs, memory telemetry, image identity, server receipts,
path counters, request outputs, matched A/B summaries, commands, revisions,
and a final `RESULTS.md`.

## 9. Stop conditions

Stop before serving A/B if:

- workspace memory prevents the fixed model from starting at utilization 0.8;
- more than one workspace appears per rank;
- M≈16K is slower than ordinary Mori at operator level;
- any rank disagrees or deadlocks;
- decode, DSpark, graph replay, or unlisted shapes enter Mega;
- output becomes non-finite or violates the accuracy gate;
- teardown leaves a container, process, core dump, or residual VRAM.

Stop before long A/B if the matched short gate does not improve P90 or causes
more than 1% ITL-tail regression.

## 10. Next step after this plan

Only after MTPR=16384 passes the matched serving gate, create:

```text
plans/06_DSV41_FLASH_MEGAMOE_AGENTX_CONFIRMATION.md
```

That plan owns the three 1200-second repetitions, canonical 3600-second
confirmation, concurrency widening, accuracy, and final KEEP/REVERT decision.
