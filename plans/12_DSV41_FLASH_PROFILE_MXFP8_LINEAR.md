# DeepSeek V4.1 Flash MXFP8 Linear Plan

> One of six profile-driven plans (09–14) from the MI355X C32 deep trace of
> 2026-09-24. The source profile, fixed baseline and performance rules are
> shared and kept once, in [Plan 09](./09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md)
> §0, §3 and §5.
> This is an internal execution document, not published InferenceX documentation.

Status: **submitted** to its own plan-executor mesh on 2026-09-25 (setup in §8)

> **Base changed (2026-09-24)** to `arbor-v2-plus-v6best-pr19` (Plan 09 §3).
> The tile-table contents and paths below were read at `93205f244`. v6best
> carries its own MXFP8 tile loader (V2 iter5), so recheck them before execution.

## 1. Objective

Reduce time in the native MXFP8 dense GEMM, the fourth-largest cost centre and
the most frequently launched kernel in the trace. No earlier plan or PR has
investigated it.

## 2. Evidence

`_mxfp8_linear_kernel`: **91,831 calls at 12.8 µs, 1178.69 ms, 10.62%** of GPU
kernel time. That is the highest call count of any kernel, about 230 per
profiler step (400 `ProfilerStep#`).

It is a Triton `tl.dot_scaled` kernel,
`vllm/model_executor/kernels/linear/mxfp8/rocm_native.py:46`, launched with the
tiles returned by `_select_cfg(M, N, K)`. Tile selection has two layers:

- **The tuned table**, `mxfp8/configs/mxfp8_linear_tiles.json`, covers six
  `(N, K)` shapes, each for M ≤ 256 in ten row buckets. A `null` entry means the
  bucket was measured and the fallback tree won.

  | (N, K) | layer, where the source names it | tuned buckets |
  | --- | --- | ---: |
  | (1792, 5120) | `fused_wqa_wkv` | 4 / 10 |
  | (1152, 5120) | | 3 / 10 |
  | (8192, 1280) | | 8 / 10 |
  | (4096, 1280) | | 10 / 10 |
  | (5120, 2048) | | 8 / 10 |
  | (5120, 576) | shared expert `down_proj`, K = 2304 / 4 | 10 / 10 |

- **The fallback**, used for M > 256 and for any other `(N, K)`, is
  `_select_cfg`'s hand-written tree. The file itself says that tree was tuned
  on MiniMax-M3, and that it missed a DSv4.1 decode shape by 1.94× before the
  table existed.

So **every prefill chunk** (up to 16,384 tokens per step) runs on untuned
tiles. The table names its generator, `eval/eval_mxfp8_tile_table.py`, but that
script is not in the tree at `93205f244`, so the table cannot currently be
regenerated.

## 3. Fixed baseline

As Plan 09 §3. Only MXFP8 tile selection and MXFP8 GEMM backend dispatch may
change.

## 4. Work items

### 4.1 Attribute

- From recorded shapes in the deep trace, histogram `(M, N, K)` against kernel
  time. Split decode (M ≤ 256) from prefill.
- Map every `(N, K)` to its layer and name the four the table leaves
  anonymous.
- Compute achieved bandwidth for decode (weight-read bound) and achieved
  TFLOP/s for prefill (compute bound), against MI355X MXFP8 peak.
- Repeat at C8 from the shared C8 capture (Plan 09 §4.1).

### 4.2 Restore the generator

Locate or rewrite `eval_mxfp8_tile_table.py` and commit it beside the table,
together with the shapes and hardware it was run on. Without it, no re-tune is
reproducible and no review can check a table entry.

### 4.3 Prefill tiles

Extend the table with M buckets above 256 for each `(N, K)`, at the chunk sizes
4.1 observes. Bench the result against a library MXFP8 GEMM for gfx950: use
whatever the image provides in hipBLASLt or AITER. Dispatch per M bucket to the
winner.

### 4.4 Decode rows at exactly C8 and C32

Verify rows are 48 at C8 and 192 at C32. For each `(N, K)`, confirm those two
buckets resolve to a measured optimum rather than a `null` defer that has not
been re-measured on this tree. Four of the six shapes have at least one `null`
bucket.

### 4.5 Launch count (secondary)

Look for separate MXFP8 GEMMs that read the same activation and could be
concatenated along N, as `fused_wqa_wkv` already is.

## 5. Gates

Numerics:

- Tile changes alter accumulation order through `BLOCK_K`. Report max absolute
  and relative error against a BF16 reference per shape.
- Any new backend runs the 1,319-example GSM8K with real DSpark block
  rejection.

Trace, on a post-change capture: MXFP8 GEMM time per profiler step falls below
the baseline 2.95 ms/step (1178.69 ms / 400), and prefill and decode are
reported separately.

Performance: Plan 09 §5. Prefill-tile work should move C32, where prefill is a
larger share of each step. C8 must not regress.

## 6. Stop conditions

- 4.1 shows both decode and prefill within about 10% of their roofline. The
  remaining headroom, about 1% of GPU time, is then below the end-to-end noise
  floor.
- A library backend wins, but its scale layout needs a conversion that costs
  more than it saves. The kernel reads plain row-major `[M, K/32]` E8M0 scales,
  not the padded, swizzled layout.

## 7. Artifacts

```text
crusoe:/home/jiaweche/dsv41-profile-20260924/12_mxfp8_linear/
```

Persist the shape histogram, the roofline table, the generator and the new
table, backend benches, error reports, the trace captures, per-replay bench
outputs and the decision.

## 8. Execution setup

Written by the operator before submission. These are facts about the
environment, not choices about the plan.

### 8.1 Node and trees

- **Node.** Job `174717` on `crsuse2-m2m-170` (`amd-burst`, 24 h, preemptible).
  It is this mesh's only node. The pool adopts jobs named `arbor-p12-*` and
  will not request another while it holds this one.
- **Job cap.** The account now holds its maximum of four jobs. If this node is
  preempted, the pool's replacement request stays queued until another job
  ends.
- **Container.** `dsv41flash_arbor` runs the image by digest (`sha256:960228cf…`).
  Weights are node-local at
  `/mnt/m2m_nobackup/jiaweche/inferencex-dsv41flash/models/DeepSeek-V4.1-Flash`,
  and the aiperf client is at `/runtime/aiperf-src`.
- **Base on the node.** The fixed base `be794db46` is overlaid on the container.
  The setup log is `$R/logs/setup.log`.
- **Not ours.**
  - Job `174441` on `m2m-016` belongs to Plan 09.
  - Job `174682` on `m2m-031` is Plan 10's node; its mesh runs on `slog-005`
    from `/home/jiaweche/Arbor-plan-executor`.
  - Job `174710` on `m2m-250` is Plan 11's node; its mesh runs on `slog-007`
    from `/home/jiaweche/Arbor-plan-executor-p11`.
  - Never touch these jobs, their containers, or those sessions. Reading their
    files is fine.
- **Paths.**
  - Artifacts (§7): `R=/home/jiaweche/dsv41-profile-20260924/12_mxfp8_linear`.
  - Base checkout: `/home/jiaweche/dsv41-merge/vllm-v6best-pr19`, a worktree of
    `/home/jiaweche/dsv41-merge/vllm`. Do not commit on it.
  - This plan's branch: make a new worktree from `be794db46`, for example
    branch `arbor-v2-plus-v6best-pr19-p12` at
    `/home/jiaweche/dsv41-merge/vllm-p12`. The `vllm-p10*` and `vllm-p11`
    worktrees belong to other plans.
- **Deploy.** Run `SRC=<worktree> WANT_SHA=<full sha> bash $R/bin/deploy_tree.sh`
  on the node.
  - It overlays `vllm/` and hash-checks **every file** in the commit against the
    container, including JSON tile tables, not only `.py` files.
  - It also confirms the compiled extensions are untouched and imports the DSA
    attention modules.
  - Anything short of `DEPLOY_OK sha=<sha>` means the tree is not what runs.
  - The MXFP8 kernel is Triton and JIT-compiled in the container, so a tile
    change needs no build step, but Triton caches compiled kernels
    node-locally.
- **Compiled changes.** Plan 10 built a hash-checked rebuild-and-install path
  for vLLM's compiled extension, in
  `/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/bin/`
  (`build_so.sh`, `deploy_so.sh`).
- **Plan 09.** Its document (§0, §3, §5 above) is
  `/home/jiaweche/dsv41-profile-20260924/plans/09_DSV41_FLASH_PROFILE_MOE_EXPERTS.md`.
- **MoE configs.** The base's PR #19 MoE-config install writes its CSV into the
  AITER package inside the container, so every server after the first runs the
  FlyDSL MoE configs. Plan 09 measured them against their absence at 0.965× at
  C32 and 1.003× at C8. They are identical in both arms of every A/B here and
  must stay so. Plan 09 owns them.
- **GPUs.** The server runs TP4 on GPUs 0–3. GPUs 4–7 are idle.

### 8.2 Existing work to reuse

- **Traces of this base, readable now.**
  - Plan 10's fresh knob-off captures of `be794db46` at C32 and C8, 400
    `ProfilerStep#` each:
    `/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/cap_base_c32/traces/`
    and `cap_base_c8/traces/`, summarised in `run1/T01_BASELINE.json`.
  - Plan 09's captures of `cbfdad177`:
    `/home/jiaweche/dsv41-profile-20260924/09_moe_experts/run1/trace_a{0,1}_c{8,32}/traces/`.
  - Only the rank-0 trace of each capture is on `/home`.
  - **Graph-replayed kernels carry no recorded shapes** in these traces
    (Plan 09 §8). Eager kernels do. Decode MXFP8 GEMMs replay inside CUDA
    graphs, so their `(M, N, K)` cannot be read from the trace. It has to
    come from elsewhere, for example from the layer structure, a temporary
    counter, or an eager capture.
- **Scripts.** Copy and change what differs:
  - Plan 09 (`/home/jiaweche/dsv41-profile-20260924/09_moe_experts/run1/bin/`):
    `capture_one.sh`, `measure_one.sh` (one A/B point: fresh control server,
    marker check, discarded warmup, one measured replay), `eval_one.sh`
    (GSM8K with real block rejection), `p09_driver.sh` and `summarize_ab.py`.
  - Plan 10 (`/home/jiaweche/dsv41-profile-20260924/10_dsa_indexer_decode/run1/bin/`):
    `dsa_attrib.py` (Kineto attribution, including graph-replayed kernels),
    `t01_driver.sh` (captures under a user unit), knob probes, and
    `t04_bench.sh` (a standalone kernel bench).
- **Decisions other meshes made on shared questions.** These include
  capture-window validity and what "knob-off" means for something already in
  the base. They are in `mesh/intake/*.decisions.jsonl` under
  `/home/jiaweche/Arbor-plan-executor/sessions/plan-executor/` (Plan 10) and
  `/home/jiaweche/Arbor-plan-executor-p11/sessions/plan-executor/` (Plan 11).
- **Harness.** Read, never edit; it belongs to another mesh's session:
  `/home/jiaweche/Arbor-dsv41-mesh/sessions/megamoe-v9/mesh/toolbox/crusoe/{boot_control_server.sh,replay_arm.sh}`.
- **The Ruby deep trace** behind §2 is on `ruby-1`, which Crusoe cannot reach,
  so §2's numbers are the reference, not a file.

### 8.3 Things that have already cost time here

- **The login node kills your background processes.**
  `/usr/local/sbin/shared-host-watch` SIGKILLs everything left in an ssh
  session's scope about a minute after the session ends, `setsid`/`nohup`
  children included, and ends every session at 8 hours. Anything that must
  outlive a command goes in a user unit:
  `systemd-run --user --collect --unit=<name> bash -lc '<cmd>'`. This mesh runs
  on `slog-006`, where lingering was enabled on 2026-09-25, so user units
  survive there. Processes the agents start inherit the keeper's unit.
- **Memory on the login node.** It caps the whole user at 4 GiB per login node,
  with throttling from 2.5 GiB, shared with all five agents of this mesh. Parse
  traces and run anything heavy on the node through
  `srun --overlap --jobid=174717`, never on the login node.
- **Broken Docker on some nodes.**
  - On `m2m-341` and `m2m-002`, `/var/lib/docker` and `/var/lib/containerd` are
    symlinks into `/mnt/m2m_nobackup` whose targets no longer exist.
  - `m2m-042` hung a three-minute smoke test.
  - Test any replacement node with `docker images` before provisioning it.
- **Stale GPU samples.** The mesh's own GPU sampling (`srun` into the node)
  fails intermittently, so recorded node state can lag by minutes.
- **Scripts written from Windows** carry CRLF. A stray `\r` turned a successful
  boot into exit 127.
- **NFS lag.** A file written on one host can read stale on another for a few
  seconds. Retry, or read it through the host that wrote it.
- **Containers write only to node-local disk.** `/home` is NFS with
  `root_squash`: write to `/mnt/m2m_nobackup` and copy durable results back.
  `/home` is about 91% full.
- **`pgrep -f` matches itself.** Inside `bash -lc "..."`, `pgrep -f` matches its
  own command line. Use the `[m]easure_one` form.
- **Heredocs.** A heredoc into `docker exec` needs `-i`.
- **Liveness.** A driver that records `STATUS=RUNNING` and then dies looks
  healthy. Check the pid or unit, not the status file.
- **Seed.** The InferenceX mount hardcodes seed 42 (Plan 09 §5).
