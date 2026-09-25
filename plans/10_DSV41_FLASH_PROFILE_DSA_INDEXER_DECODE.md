# DeepSeek V4.1 Flash DSA Indexer Decode Plan

> One of six profile-driven plans (09–14) from the MI355X C32 deep trace of
> 2026-09-24. The source profile, fixed baseline and performance rules are
> shared and kept once, in [Plan 09](./09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md)
> §0, §3 and §5.
> This is an internal execution document, not published InferenceX documentation.

Status: **submitted** to the plan-executor mesh on 2026-09-25 (setup in §8)

> **Base changed (2026-09-24)** to `arbor-v2-plus-v6best-pr19` (Plan 09 §3).
> The code references below were read at `93205f244` and must be rechecked on
> the new base before execution:
>
> - PR #19 items 1 and 3 are now in the base.
> - v6best has no VARCTX schedule, so the `WavePerEU` desynchronisation in §2.2
>   does not arise there.
> - This plan's question becomes whether those two items help, measured against
>   their own absence.

## 1. Objective

Cut the time the DSA indexer spends choosing which KV positions each decode row
attends to (paged MQA logits, then top-k), at both C8 and C32, without changing
which positions are selected.

## 2. Evidence

DSA attention plus top-k is about 18.8% of GPU kernel time. This plan owns the
indexer part of it:

| kernel | calls | ms | µs/call | share |
| --- | ---: | ---: | ---: | ---: |
| `topKPerRowDecode`, 3 variants | 4,812 | 622.17 | 262.4 / 95.3 / 30.2 | 5.60% |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx` | 3,208 | 443.55 | 138.3 | 3.99% |
| `_onepass_topk_csr` | 13,110 | 134.76 | 10.3 | 1.21% |

Each `topKPerRowDecode` variant has 1,604 calls. The slow one,
`topKPerRowDecode<512, true, true, false>`, costs 420.9 ms at 262.4 µs, which is
expensive for a top-k. The server log contains `DSA decode top-k: native kernel`
**26,777** times, including for `compress_ratio=1`.

Out of scope here: `_sparse_attn_decode_gfx950_partial` (3.62%),
`_sparse_attn_decode_reduce` (0.88%) and the prefill logits kernels.

### 2.1 Why decode top-k lands on the native kernel

In `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` at `93205f244`:

- `_get_aiter_top_k_kernel` (line 583) returns `None`, meaning native, for
  `compress_ratio <= 1`. V4.1 layers use ratios 0, 1 and 2
  (`vllm/models/deepseek_v4_1/attention.py:245`), so ratio-1 layers never reach
  AITER.
- For ratio-2 decode, the base 1-D policy chooses native when
  `num_rows <= _GFX950_C4A_NATIVE_MAX_ROWS` (256) and
  `max_valid_seq_len > _GFX950_C4A_AITER_MAX_COMPRESSED_SEQ_LEN` (64 × 1024).
  Its comment gives the reason: AITER v0.1.19 decode top-k is one-block only.
- Under FULL CUDA-graph replay, the caller passes `max_model_len` as the context
  bound, because graphs are not keyed by context length. With
  `--max-model-len 1048576`, every FULL-replayed decode batch of at most 256 rows
  takes the native kernel whatever its real context is. C8 verify batches are
  48 rows and C32 batches are 192, so both qualify.
- The optional 2-D policy (`VLLM_ROCM_C4A_TOPK_SHAPE_POLICY`, default 0) also
  resolves to native at 1M, as its own comment says.

Real contexts are long too: the deepest observed was 718,125 compressed tokens.
Where a batch's true maximum context exceeds 64K, native is the correct choice
for the current AITER kernel anyway. The addressable share is therefore
ratio-2 batches whose true maximum is at most 64K, plus ratio-1 layers wherever
AITER proves correct. Step 4.1 measures how big that is.

### 2.2 PR #19 against this tree

[PR #19](https://github.com/qianghan-amd/vllm-eed1f3d0/pull/19) (open) has two
items here, and neither is in `93205f244`:

- **Item 1** changes the ratio test to `compress_ratio == 0` and skips the
  native-max-rows gate for ratio 1. That sends ratio-1 layers to AITER at any
  context, including the range above 64K that the one-block limit guards.
  Correctness there is unproven. The PR's measured baseline already includes
  this item, so its speed contribution has never been measured.
- **Item 3** raises `WavePerEU` from 2 to 4 at the `deepgemm_fp8_paged_mqa_logits`
  call (line 1319). On this tree `vllm/v1/attention/backends/mla/indexer.py:686`
  pins `_AITER_VARCTX_WAVE_PER_EU = 2` for the VARCTX schedule, with the comment
  that it must match the call site "or the schedule solves a different problem
  than the kernel executes". VARCTX is on by default here. PR #19 touches only
  `rocm_aiter_mla_sparse.py` and `rocm_fp32_router_gemm.py`, so porting item 3
  as written would desynchronise the schedule from the kernel.

## 3. Fixed baseline

As Plan 09 §3. Only indexer decode top-k selection and the paged-MQA-logits
launch parameters may change.

## 4. Work items

### 4.1 Attribute

- Map the three `topKPerRowDecode` variants to compress ratio and phase, using
  template arguments and recorded shapes.
- Record the per-batch **true** maximum compressed context for decode. Use a
  temporary counter, not log lines: these markers are `logger.info_once`, and
  the first line is the least informative one for a batch-dependent choice.
- Report the fraction of decode top-k calls that are ratio 1, ratio 2 at most
  64K, and ratio 2 above 64K, at C32 from the deep trace and at C8 from the
  shared C8 capture (Plan 09 §4.1).

### 4.2 Correctness and speed, standalone

Bench native against AITER top-k at 48 and 192 rows across compressed contexts
from 4K to 1M, including above 64K for ratio 1. Compare selected index sets
with `torch.topk`, allowing permutations among tied scores. AITER is admissible
only where the sets match.

### 4.3 Candidates, one knob each, default off

- **A. Ratio-1 routing** (PR #19 item 1), limited to the context range where
  4.2 shows AITER correct.
- **B. Context-aware choice under FULL graphs.** Decide on the device from the
  real `seq_lens` inside one launch, or capture per context bucket, so replay no
  longer needs the `max_model_len` bound.
- **C. A faster native kernel** for the 262.4 µs variant
  (`csrc/libtorch_stable/topk.cu`, `sampler.cu`). This helps even the batches
  that must stay native.
- **D. Logits waves** (PR #19 item 3), with `_AITER_VARCTX_WAVE_PER_EU` changed
  in lockstep and the pair asserted equal at import.

## 5. Gates

Correctness: index-set match against the reference on every admitted shape and
context range. This is a hard gate, because a wrong top-k changes attention
without any error.

Trace, on a post-change capture:

- the native-kernel share of decode top-k falls by the fraction predicted in
  4.1;
- `topKPerRowDecode` time per profiler step falls below the baseline
  1.56 ms/step (622.17 ms / 400 `ProfilerStep#`);
- for D, the VARCTX logits kernel falls below 138.3 µs/call, the PR claims by
  7–10%.

Performance: Plan 09 §5. Report C8 separately. On B200 the sparse-attention
indexer carries no time at all in the C8 decode window, so the MI355X C8 share
must be measured, not assumed.

## 6. Stop conditions

- 4.1 shows the addressable share is small at both points. Only C then remains
  worth doing.
- AITER disagrees with the reference in the range a candidate needs.
- D cannot keep schedule and kernel constants equal.

## 7. Artifacts

```text
crusoe:/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/
```

Persist the context histograms, the index-set comparison, the kernel bench,
the knob diffs, the trace captures, per-replay bench outputs and the decision.

## 8. Execution setup

Written by the operator before submission. These are facts about the
environment, not choices about the plan.

### 8.1 Node and trees

- **Node.** Job `174682` on `crsuse2-m2m-031` (`amd-burst`, 24 h, preemptible).
  It is the executor mesh's only node, and the pool will not request another
  while it is held.
- **Container.** `dsv41flash_arbor` runs the image by digest (`sha256:960228cf…`).
  Weights are node-local at
  `/mnt/m2m_nobackup/jiaweche/inferencex-dsv41flash/models/DeepSeek-V4.1-Flash`,
  and the aiperf client is at `/runtime/aiperf-src`.
- **Base on the node.** The fixed base `be794db46` is overlaid on the container.
  The setup log is `$R/logs/setup.log`.
- **Not ours.** Job `174441` on `m2m-016` is Plan 09's A/B. Never touch it or
  its container.
- **Paths.**
  - Artifacts (§7): `R=/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode`.
  - Base checkout: `/home/jiaweche/dsv41-merge/vllm-v6best-pr19`, a worktree of
    `/home/jiaweche/dsv41-merge/vllm`. Do not commit on it.
  - This plan's branch: make a new worktree from `be794db46`, for example
    branch `arbor-v2-plus-v6best-pr19-p10` at
    `/home/jiaweche/dsv41-merge/vllm-p10`.
- **Deploy.** Run `SRC=<worktree> WANT_SHA=<full sha> bash $R/bin/deploy_tree.sh`
  on the node. It overlays `vllm/`, hash-checks every `.py` in the commit
  against the container, confirms the compiled extensions are untouched, and
  imports the DSA attention modules. Anything short of `DEPLOY_OK sha=<sha>`
  means the tree is not what runs.
- **Plan 09 document.** Plan 09 (§0, §3, §5 referenced above) is
  `/home/jiaweche/dsv41-profile-20260924/plans/09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md`.
- **AITER version.** The image's AITER is 0.1.21.post2. The one-block
  limitation cited in §2.1 was written against v0.1.19, so 4.2 tests the
  installed version.
- **MoE configs.** The base's PR #19 MoE-config install writes its CSV into the
  AITER package inside the container, so every server after the first runs the
  FlyDSL MoE configs. That is identical in both arms of every A/B here and must
  stay so. Plan 09 owns it.

### 8.2 Plan 09 is the working template

`/home/jiaweche/dsv41-profile-20260924/09_moe_experts/` did the same kind of
work on the same base, and its scripts are known to work. Copy them and change
what differs:

- `run1/bin/capture_one.sh` + `boot_variant.sh`: one profiled capture
  (`active_iterations=400`). The window starts by elapsed time, because AgentX
  occupancy sits at 1–6 for long stretches and a steady-state trigger never
  fires.
- `run1/bin/measure_one.sh`: one A/B point. It boots a fresh control server
  with the arm's env file, checks the knob marker, runs a discarded warmup and
  then one measured replay.
- `run1/bin/eval_one.sh`: GSM8K with real block rejection.
- `run1/bin/p09_driver.sh`: sequencing, skip-if-`DONE`, staging node-local
  results to `/home`, and alternating arm order between rounds.
- `run1/bin/moe_attrib.py`: Kineto parsing, including attribution of
  graph-replayed kernels through the `correlation` of their `hipGraphLaunch`.
  Also `summarize_ab.py` and `trace_gate.py`.
- Harness (read, never edit; it belongs to another mesh's session):
  `/home/jiaweche/Arbor-dsv41-mesh/sessions/megamoe-v9/mesh/toolbox/crusoe/{boot_control_server.sh,replay_arm.sh}`.

**Traces.** Plan 09's captures were lost with `m2m-037`, and the Ruby deep
trace behind §2 is on `ruby-1`, which Crusoe cannot reach. 4.1 therefore needs
fresh knob-off captures of the base at C32 and C8 on this node.

### 8.3 Things that have already cost time here

- **The login node kills your background processes.** Since 2026-09-25 06:39
  UTC, `/usr/local/sbin/shared-host-watch` SIGKILLs everything left in an ssh
  session's scope about a minute after the session ends. That includes
  `setsid`/`nohup` children. It also ends every session at 8 hours. Anything
  that must outlive a command goes in a user unit:
  `systemd-run --user --collect --unit=<name> bash -lc '<cmd>'`. Check it with
  `systemctl --user status <name>`. It killed Plan 09's driver twice and a node
  step with it.
- **Memory on the login node.** It caps the whole user at 4 GiB, with
  throttling from 2.5 GiB, shared with all five mesh agents. Parse traces and
  run anything heavy on the node through `srun --overlap --jobid=174682`, never
  on the login node.
- **Broken Docker on some nodes.** On `m2m-341` and `m2m-002`,
  `/var/lib/docker` and `/var/lib/containerd` are symlinks into
  `/mnt/m2m_nobackup` whose targets no longer exist, so every image operation
  fails. `m2m-042` hung a three-minute smoke test. If this node is preempted,
  test a replacement with `docker images` before provisioning it.
- **Scripts written from Windows** carry CRLF. A stray `\r` turned a successful
  boot into exit 127.
- **NFS lag.** The login node sees a `STATUS` file the node just wrote on
  `/home` late. Retry, or read it through the node.
- **Containers write only to node-local disk.** `/home` is NFS with
  `root_squash`: write to `/mnt/m2m_nobackup` and copy durable results back.
- **`pgrep -f` matches itself.** Inside `bash -lc "..."`, `pgrep -f` matches its
  own command line. Use the `[m]easure_one` form.
- **Heredocs.** A heredoc into `docker exec` needs `-i`.
- **Liveness.** A driver that records `STATUS=RUNNING` and then dies looks
  healthy. Check the pid or unit, not the status file.
- **Seed.** The InferenceX mount hardcodes seed 42 (Plan 09 §5).
