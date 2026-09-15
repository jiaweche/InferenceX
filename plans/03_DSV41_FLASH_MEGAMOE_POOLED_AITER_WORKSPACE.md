# DeepSeek V4.1 Flash Pooled AITER Workspace and Teardown Plan

> Focused execution plan under the
> [DeepSeek V4.1 Flash MegaMoE + AgentX master plan](./01_DSV41_FLASH_MEGAMOE_AGENTX.md).
> The validated starting evidence is recorded in
> [the Ruby handoff](./02_DSV41_FLASH_MEGAMOE_RUBY_HANDOFF.md).
> This is an internal execution document, not published InferenceX documentation.

Status: **completed on September 15, 2026** at AITER
`e4b600af4ed97453a325711e2fc2aee0ebeb5239`.

## 1. Objective

Make AITER MegaMoEV2 safe to reuse across all 40 DeepSeek V4.1 backbone MoE
layers without copying expert weights or allocating one multi-gigabyte
communication workspace per layer.

This step must also provide deterministic, idempotent teardown for every
MegaMoEV2-owned local and MoRI symmetric allocation before communicator and
process-group destruction.

Completion of this plan proves an operator lifecycle primitive. It does not
enable MegaMoEV2 in vLLM or establish an end-to-end performance gain.

## 2. Starting state and fixed provenance

Use the retained Ruby allocation on `cv350-rck-g03-c09-18` and these working
locations:

```text
InferenceX plan/worktree:
  /scratch/jiaweche/inferencex-dsv41flash/repo-continuation

AITER source:
  /scratch/jiaweche/dsv41-megamoe/aiter-vllm
  base 797cce253bbadbaf651cdd93527806aa989ea56b

MoRI source:
  /scratch/jiaweche/dsv41-megamoe/mori-minimal
  base 0a1cd437317463f96906f20ca0566e4e969d8471

Pinned vLLM reference:
  /scratch/jiaweche/dsv41-megamoe/vllm-src
  eed1f3d0c6043bd494424a22443ee198dd56f657

Durable artifacts:
  /home/jiaweche/dsv41-megamoe-validation-20260915
```

The validated MoRI library with borrowed external-handle ownership has SHA256:

```text
3df6da1342f1c9dc7923fd2620bb132b283b2063bf0040a88cb08056e136cfd5
```

Existing evidence already proves:

- captured backbone M=6 and DSpark M=5 against the independent Torch oracle;
- captured prefill M=888 against the independent Torch oracle;
- graph replay for captured M=6, 5, 888, 2625, and 7100;
- zero-token and uneven-rank operation;
- 20 repeated bursts;
- maximum configured MTPR 16384;
- clean one-, four-, and eight-rank teardown with the MoRI ownership fix.

## 3. Why pooling is required

`MegaMoEV2` currently binds one layer's `w1`, `w1_scale`, `w2`, and
`w2_scale` in its constructor and allocates all dispatch, Stage1, Stage2,
quantization, and combine buffers per instance.

The initial static estimate understated several symmetric buffers. The final
TP4 measurement at MTPR 8192 is 2,236,754,288 bytes per rank: 1,695,265,648
symmetric bytes plus 541,488,640 CUDA-managed bytes. Instantiating one for
every backbone MoE layer would therefore consume approximately 89.5 GB per
rank, including approximately 67.8 GB of repeated symmetric allocations.

The observed candidate prefill values are M=888, 2625, and 7100. Therefore:

- use one pooled workspace per rank;
- size it to the next power of two above the largest proven candidate:
  MTPR 8192;
- reuse the finalized, already shuffled local expert weights by pointer;
- retain the ordinary MTPR 16384 path as the fallback outside this workspace.

## 4. Scope

This plan includes:

1. A backward-compatible weight-explicit MegaMoEV2 execution API.
2. One workspace reused serially across different layer weights.
3. Strict no-copy weight-layout validation.
4. Explicit ownership tracking for every symmetric allocation.
5. Idempotent close and close/recreate behavior.
6. Distributed correctness, graph, burst, memory, and performance gates at
   MTPR 8192.
7. Durable command, revision, checksum, and result receipts.

This plan excludes:

- the vLLM runtime selector;
- DeepSeek model or MXFP4 method changes;
- shared-expert scheduling inside vLLM;
- server startup, AgentX, or accuracy runs;
- changing the decode or DSpark fallback policy;
- EPLB, multi-node, DP>1, PP>1, or sequence-parallel support.

## 5. Design invariants

The implementation must satisfy all of the following:

- Exactly one MegaMoEV2 communication workspace exists for the V4.1 backbone
  on each rank.
- At most one launch is in flight on that workspace.
- All 40 layers may supply different weight pointers without reallocating the
  workspace.
- Weight tensors remain owned by the model. The workspace borrows them only
  for a call and never frees or replaces them.
- The weight-explicit path rejects non-contiguous or incompatible layouts. It
  must not call `.contiguous()` and silently create a layer-sized copy.
- Existing constructor-bound callers continue to work through a compatibility
  wrapper.
- Every symmetric tensor is freed exactly once and in reverse allocation
  order.
- No collective, barrier, or free occurs from `__del__`; teardown is explicit.
- Every rank enters allocation, execution, and teardown in the same order.
- Calling `close()` more than once is safe.
- Calls after `close()` fail clearly before launching a kernel.

## 6. AITER implementation

### 6.1 Add a weight-explicit execution path

Modify:

```text
aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
```

Add a `forward_with_weights` path accepting:

```text
x_bf16
topk_weights
topk_ids
w1
w1_scale
w2
w2_scale
```

Thread these weight tensors explicitly through Stage1 and Stage2 launch
helpers instead of reading only constructor-bound attributes.

The existing `forward` API remains and delegates to `forward_with_weights`
using the constructor-bound weights. This preserves existing callers and
keeps the refactor independently reviewable.

Validate before launch:

- local expert count is 96 for the V4.1 EP4 case;
- packed weight and scale shapes match the runner geometry;
- weights and scales are contiguous;
- dtypes and shuffle layout markers match AITER A16W4;
- all tensors are on the runner's device;
- no tensor aliases an owned workspace allocation.

The validation path must return no transformed tensor. Pointer identity before
and during the call is part of the test contract.

### 6.2 Preserve asynchronous pointer safety

Kernel launch arguments capture device pointers by value. The implementation
must still prove that alternating two layer-weight sets cannot race through
mutable Python attributes.

Prefer explicit arguments throughout the launch stack. If a temporary
binding helper is required internally:

- bind immediately before Stage1;
- launch Stage1 and Stage2 on the ordered streams;
- preserve the existing residual-stream event dependency;
- prevent a second call until the first call's workspace use is ordered;
- clear borrowed references during `close()`.

Do not add a global process-wide weight singleton.

### 6.3 Track all owned symmetric allocations

Modify:

```text
aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py
```

Both `FlyDSLDispatchCombineIntraNodeOp` and
`FlyDSLDispatchGroupMajorOp` must record every tensor returned by
`mori_shmem_create_tensor`.

Allocation helpers must:

1. allocate;
2. append the original owning tensor to an ordered ownership list;
3. create views and P2P pointer tables only after ownership is recorded.

Views, pointer tables, integer pointer wrappers, and borrowed model weights are
not owners and must never be passed to `mori_shmem_free_tensor`.

### 6.4 Add explicit teardown

Add idempotent `close()` methods to:

```text
FlyDSLDispatchGroupMajorOp
FlyDSLDispatchCombineIntraNodeOp
MegaMoEV2
```

Use this order:

1. reject new launches;
2. synchronize the active device and residual stream;
3. enter a rank-symmetric MoRI barrier;
4. clear kernel pointer tables, views, events, and borrowed weight references;
5. close the group-major child;
6. free each original symmetric tensor in reverse allocation order;
7. clear local GPU workspaces and compiled-object references;
8. enter a final rank-symmetric barrier;
9. mark the object closed.

If teardown raises, retain enough state to avoid a double free on a retry and
report which allocation failed. Do not hide failures in the explicit close
path.

Add context-manager support only as a thin wrapper around explicit `close()`.
Do not rely on garbage collection for correctness.

## 7. Tests and evidence

### 7.1 Focused API tests

Add or extend AITER tests to cover:

- constructor-bound `forward` remains behaviorally unchanged;
- `forward_with_weights` rejects wrong device, shape, dtype, layout, and
  contiguity;
- a plausible non-contiguous input raises instead of copying;
- `close()` is idempotent;
- a post-close call raises;
- close followed by a new allocation succeeds.

Tests must execute behavior and assert outputs or failures. They must not read
or grep source text.

### 7.2 TP4 alternating-weight correctness

On four gfx950 GPUs:

1. Allocate one MTPR 8192 workspace.
2. Create two independently seeded local expert-weight sets in the same
   validated packed layout.
3. Alternate A, B, A, B across at least 20 calls using captured M=888 and
   M=7100 routes.
4. Compare small/medium cases with the independent Torch oracle.
5. Verify the second A result matches the first A result within the existing
   `relL2 <= 0.10` gate.
6. Verify graph replay remains exact where graph capture is enabled.

This test detects stale pointers, cross-layer workspace contamination, and
incorrect asynchronous rebinding.

### 7.3 Forty-layer reuse and memory gate

Simulate 40 layer bindings across two independently seeded full weight sets
without allocating 40 workspaces.

Record per rank:

- memory before workspace creation;
- memory after one workspace;
- memory after all 40 layer bindings;
- peak allocated and reserved memory;
- MoRI symmetric allocation count and bytes;
- model-weight storage pointers before and after every bind.

Acceptance:

- one workspace allocation sequence per rank;
- no growth proportional to layer count;
- combined symmetric and CUDA-managed workspace no greater than 2.4 GB
  decimal per rank at MTPR 8192;
- zero additional expert-weight storage;
- all original weight storage pointers unchanged.

### 7.4 Captured-shape regression

Run MTPR 8192 with the durable route corpora:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/agentx/shape-capture-v3/routes/
/home/jiaweche/dsv41-megamoe-validation-20260915/agentx/shape-capture-v3/routes-count-exact/
```

Cover:

- M=888 against the independent Torch oracle;
- M=2625 and M=7100 graph replay;
- the count-exact layer-10 M=7100 skew;
- zero-token and uneven ranks;
- at least 20 alternating bursts;
- close/recreate after the burst.

Repeat the equal-topology M=888, M=2625, and hot-skew M=7100 operator A/B
three times. M=7100 must retain at least a 10% rank-maximum advantage over the
MoRI plus ordinary-AITER control. Decode M=6 and DSpark M=5 remain fallback
evidence and are not promotion targets.

### 7.5 Teardown gate

Run explicit close on one, four, and eight ranks where the shape permits.

Require:

- every rank prints a post-close sentinel;
- process exit code zero;
- no SIGSEGV, core dump, HIP error, or watchdog timeout;
- no residual container;
- zero VRAM after process exit;
- a second process can initialize, execute, close, and exit on the same GPUs.

## 8. Artifact contract

Write durable evidence under:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/operator/pooled-workspace/
```

Persist:

- `revisions.txt`;
- exact command receipts;
- build logs and binary SHA256;
- focused test logs;
- TP4 and TP8 rank logs;
- memory summaries in JSON;
- allocation-count summaries in JSON;
- three-run A/B summaries in JSON;
- final `RESULTS.md` stating pass, fail, or blocked for every gate.

Do not store checkpoints, image layers, build trees, or compiler caches there.
Keep those under `/scratch/jiaweche/dsv41-megamoe/`.

Before releasing the allocation, verify every artifact is readable from a
fresh shell and regenerate the checked-in AITER patch from the exact tested
source revision.

## 9. Completion and stop conditions

This plan is complete only when:

- the weight-explicit API is backward compatible;
- one workspace safely alternates distinct layer weights;
- MTPR 8192 passes the captured-shape gates;
- memory remains constant across 40 bindings;
- no model-weight copy is created;
- explicit close/recreate passes on all tested ranks;
- final source, binary, commands, and evidence agree;
- the AITER change is committed and pushed to its designated branch;
- this master-plan child is updated with final revisions and results.

Stop and diagnose if:

- a layer requires its own symmetric workspace;
- correctness depends on synchronizing after every layer;
- explicit weight arguments cause a hidden contiguous copy;
- workspace reuse changes route or output semantics;
- close can deadlock when all ranks call it;
- any allocation lacks a single identifiable owner;
- MTPR 8192 removes the measured prefill advantage;
- memory or lifecycle evidence cannot be reproduced from the recorded command.

## 10. Next step after this plan

Every completion gate above passed. The successor is
[plan 04: selective vLLM MegaMoEV2 adapter](./04_DSV41_FLASH_MEGAMOE_VLLM_ADAPTER.md).

That child plan will implement the prefill-only, fail-closed vLLM adapter:

- branch after the existing router in `Mxfp4MoEMethod.apply`;
- borrow finalized AITER MXFP4 weights without copying;
- let `MoriAll2AllManager` own exactly one pooled MegaMoEV2 workspace;
- retain existing `MoERunner` shared-expert overlap and reduction behavior;
- use `ForwardContext` to permit only pure eager prefill M=888, 2625, or 7100;
- require one cached rank-consistent decision per model forward;
- fall back for decode, DSpark, mixed/unknown phases, graphs, and unsupported
  topology;
- run a matched EP4 server smoke, 60-second replay, AgentX A/B, and accuracy
  gates.

Do not begin the vLLM adapter merely because the pooled API compiles. Advance
only from the recorded correctness, memory, performance, and teardown
evidence.

## 11. Execution results

Result: **PASS**

Source:

```text
AITER base:
  797cce253bbadbaf651cdd93527806aa989ea56b
AITER result:
  e4b600af4ed97453a325711e2fc2aee0ebeb5239
AITER patch SHA256:
  0b90b1dac1cb82d751fd97b1ea0dcac6a338c89235a729e61b008531dbcf2e8b
MoRI library SHA256:
  3df6da1342f1c9dc7923fd2620bb132b283b2063bf0040a88cb08056e136cfd5
```

The implementation:

- adds strict weight-explicit execution while preserving constructor-bound
  callers;
- retains borrowed weight tensors for asynchronous and graph lifetime without
  copying storage;
- rejects non-contiguous or unmarked explicit-path weights;
- passes different W1/W2 pointers through both stages and the residual stream;
- tracks 39 original symmetric allocations;
- closes on the runner's own device, independent of the ambient device;
- frees symmetric allocations in reverse order;
- supports idempotent close, post-close rejection, and close/recreate;
- closes each superseded runner in multi-size tests.

Final TP4 M=888 validation alternated two independently seeded weight sets 40
times. Both references passed at `relL2=0.058254/0.058279`; A-to-B-to-graph-A
replay was exact; weight pointers were unchanged; and steady allocated-memory
growth was zero.

Measured TP4 MTPR 8192 workspace per rank:

```text
symmetric allocations:      39
symmetric bytes:            1,695,265,648
CUDA-managed delta:           541,488,640
combined workspace bytes:   2,236,754,288
```

The same workspace passed captured M=2625, count-exact hot-skew M=7100,
maximum M=8192, zero/uneven ranks, multi-size replacement, TP1 DSpark
close/recreate, and TP8 V4.1 close/recreate. The final TP8 communication-fused
regression also exited cleanly.

Three-run rank-maximum medians at MTPR 8192:

```text
M=888:       control 1.4244 ms, Mega 0.8643 ms, +64.74%
M=2625:      control 2.5460 ms, Mega 1.9103 ms, +33.29%
M=7100 hot:  control 6.0151 ms, Mega 5.0331 ms, +19.48%
```

Durable evidence:

```text
/home/jiaweche/dsv41-megamoe-validation-20260915/operator/pooled-workspace/
```

Rank-local allocator failure recovery remains fail-stop, and the standalone
benchmark's exceptional path does not yet guarantee close. These are recorded
limitations, not failures of the validated normal lifecycle required by the
next adapter step.
