#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash on MI355X: native DSpark and GPU-resident KV.
# Follow upstream AMD defaults for Engram; storage behavior needs verification.
# Image: vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657
# MI355X run 34710937012 passed concurrency 1-32 and eval-only concurrency 32.
# https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

# Complete/resume partial downloads instead of trusting nonempty directories.
MODEL_DOWNLOAD_ARGS=("$MODEL")
if [[ -n "${MODEL_REVISION:-}" ]]; then
    MODEL_DOWNLOAD_ARGS+=(--revision "$MODEL_REVISION")
fi
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "${MODEL_DOWNLOAD_ARGS[@]}" --local-dir "$MODEL_PATH"
else
    hf download "${MODEL_DOWNLOAD_ARGS[@]}"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
# AITER's Triton MoE GEMM repeatedly warns that Gluon is unavailable and falls
# back to Triton. Gluon supports only gfx1250, so on this gfx950 recipe that
# message was 98% of a gsm8k server log (411k of 417k lines, 30 MiB of 32 MiB).
# This process-global threshold can also hide other AITER Triton warnings; set
# it back to WARNING while diagnosing new startup or runtime failures. This is
# log hygiene only: the observed emits cost 0.03% of wall time per worker.
export AITER_TRITON_LOG_LEVEL=ERROR
# DeepseekV41ForCausalLM is not torch-compiled upstream, so the default
# cudagraph_mode=FULL_AND_PIECEWISE aborts at engine init with "piecewise CUDA
# graphs unavailable" (run 34566727564). The model is built for the breakable
# cudagraph path -- amd/attention.py uses eager_break_during_capture.
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_NUM_THREADS=1
shape_capture=0
case "${VLLM_MOE_SHAPE_CAPTURE:-0}" in
    1|true|TRUE|yes|YES)
        shape_capture=1
        export VLLM_USE_BREAKABLE_CUDAGRAPH=0
        export VLLM_MOE_SHAPE_CAPTURE_PATH="${VLLM_MOE_SHAPE_CAPTURE_PATH:-/results/moe_shape_capture.jsonl}"
        export VLLM_MOE_SHAPE_CAPTURE_ACTIVE_FILE="${VLLM_MOE_SHAPE_CAPTURE_ACTIVE_FILE:-/results/moe_shape_capture.active}"
        rm -f \
            "$VLLM_MOE_SHAPE_CAPTURE_PATH" \
            "$VLLM_MOE_SHAPE_CAPTURE_ACTIVE_FILE"
        ;;
esac
aiter_mega_moe_v2=0
force_mori_all2all=0
case "${VLLM_AITER_MEGA_MOE_V2:-0}" in
    1|true|TRUE|yes|YES)
        aiter_mega_moe_v2=1
        force_mori_all2all=1
        export AITER_MEGA_MOE_PRELOAD=1
        ;;
esac
case "${VLLM_FORCE_MORI_ALL2ALL:-0}" in
    1|true|TRUE|yes|YES)
        force_mori_all2all=1
        ;;
esac
if (( force_mori_all2all )); then
    export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-40G}"
fi
# Pin the full-context corpus for this 1M-context recipe.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

# Explicit reproducibility cap. Upstream vllm serve selects 1024 on GPUs with
# at least 160 GiB, while the previous local 2*CONC cap sat below AgentX's
# subagent fan-out. At CONC=1 it admitted 2 requests and left the rest queued
# on scheduling capacity. Pinning 128 also keeps CAPTURE_SIZE deterministic.
MAX_NUM_SEQS=128
NUM_SPEC_TOKENS=5
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < MAX_NUM_SEQS * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done
if (( force_mori_all2all )); then
    CAPTURE_SIZE="${VLLM_MEGAMOE_CUDAGRAPH_CAPTURE_SIZE:-128}"
fi

# Use the runner-specific port assigned by launch_mi355x-amds.sh.
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# Golden AL: golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Accuracy evals keep real block rejection; throughput fixes acceptance to AL 3.51.
# Adaptive verification stays off in both modes on ROCm: it trims verification
# requests on device, which DeepseekV4IndexerBackend does not support, so the
# engine refused to start with it enabled (run 34651830283, eval-only c32).
if [[ "${EVAL_ONLY:-false}" == true ]]; then
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
else
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
fi
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP"
    --language-model-only
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    # aiter, not aiter_triton_mxfp4_bf16: the plain name opens vLLM's full
    # priority list and the CK kernel at its head wins. Despite the BF16
    # backend name and this checkpoint's activation_scheme=dynamic, CK
    # quantizes activations to FP8 internally and dispatches the a8w4 experts
    # (mfma_moe1_silu_mul_afp8_wfp4_bf16 / mfma_moe2_afp8_wfp4_bf16) that the
    # DSV4-Pro MI355X recipe already gets. Pinning the Triton name instead
    # forced the W4A16 _moe_gemm_a16w4 kernel.
    --moe-backend aiter
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
    --speculative-config "$SPEC_CONFIG"
    --max-model-len 1048576
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --max-num-batched-tokens 16384
    --disable-uvicorn-access-log
)
if (( force_mori_all2all )); then
    VLLM_CMD+=(
        --enable-expert-parallel
        --all2all-backend mori_high_throughput
        # The full 16,384-token compile warmup faults inside sparse attention
        # before serving. Prefill remains eager; decode retains full graphs.
        --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
    )
    if (( aiter_mega_moe_v2 )); then
        VLLM_CMD+=(
            --kernel-config
            '{"enable_aiter_mega_moe_v2":true,"aiter_mega_moe_v2_max_tokens":16384,"aiter_mega_moe_v2_token_allowlist":[888,889,2625,2626,7100,7101,16376,16380]}'
        )
    else
        VLLM_CMD+=(--kernel-config '{"force_mori_all2all":true}')
    fi
fi
if (( shape_capture )); then
    # Diagnostic only: Python must observe every real routing invocation.
    VLLM_CMD+=(--enforce-eager)
fi
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
SERVER_PID=""
SHAPE_CAPTURE_WATCHER_PID=""
cleanup_server() {
    local rc=$?
    trap - EXIT INT TERM
    if (( shape_capture )); then
        [[ -z "$SHAPE_CAPTURE_WATCHER_PID" ]] ||
            kill "$SHAPE_CAPTURE_WATCHER_PID" 2>/dev/null || true
        rm -f "$VLLM_MOE_SHAPE_CAPTURE_ACTIVE_FILE"
    fi
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$rc"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY:-false}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    if (( shape_capture )); then
        (
            while kill -0 "$SERVER_PID" 2>/dev/null; do
                if [[ -f "$RESULT_DIR/benchmark.log" ]] &&
                   [[ "$(< "$RESULT_DIR/benchmark.log")" == *"Phase profiling (profiling) started"* ]]; then
                    : > "$VLLM_MOE_SHAPE_CAPTURE_ACTIVE_FILE"
                    exit 0
                fi
                sleep 1
            done
            exit 1
        ) &
        SHAPE_CAPTURE_WATCHER_PID=$!
    fi
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
    if (( shape_capture )); then
        [[ -z "$SHAPE_CAPTURE_WATCHER_PID" ]] ||
            kill "$SHAPE_CAPTURE_WATCHER_PID" 2>/dev/null || true
        SHAPE_CAPTURE_WATCHER_PID=""
        rm -f "$VLLM_MOE_SHAPE_CAPTURE_ACTIVE_FILE"
    fi
fi
