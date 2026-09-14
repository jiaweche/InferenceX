#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Pro AgentX preset for an already allocated Ruby MI350X worker.
# The underlying benchmark remains the checked-in MI355X recipe because both
# SKUs use gfx950; results must still be labeled as MI350X.

export IMAGE="${IMAGE:-lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260822}"
export MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
export MODEL_PREFIX="${MODEL_PREFIX:-dsv4}"
export PRECISION="${PRECISION:-fp4}"
export FRAMEWORK="${FRAMEWORK:-sglang}"
export SPEC_DECODING="${SPEC_DECODING:-mtp}"

export TP="${TP:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export DCP_SIZE="${DCP_SIZE:-1}"
export PCP_SIZE="${PCP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-1}"
export DP_ATTENTION="${DP_ATTENTION:-false}"
export CONC="${CONC:-1}"

if [[ "$TP" == "8" ]]; then
    export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
else
    export KV_OFFLOADING="${KV_OFFLOADING:-none}"
fi

case "$KV_OFFLOADING" in
    none)
        unset KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA || true
        export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-0}"
        kv_tag="kvnone"
        ;;
    dram)
        export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
        if [[ "$KV_OFFLOAD_BACKEND" != "hicache" ]]; then
            echo "Error: the DeepSeek V4 SGLang recipe supports only HiCache DRAM offload" >&2
            exit 1
        fi
        export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\":\"hicache\"}}"
        export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-2399}"
        kv_tag="kvdram-hicache"
        ;;
    *)
        echo "Error: KV_OFFLOADING must be 'none' or 'dram', got '$KV_OFFLOADING'" >&2
        exit 1
        ;;
esac

export EXP_NAME="${EXP_NAME:-dsv4_tp${TP}_conc${CONC}_${kv_tag}_spec-mtp}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"
export ENABLE_AGENTX_POWER="${ENABLE_AGENTX_POWER:-0}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export RUBY_SCRATCH_ROOT="${RUBY_SCRATCH_ROOT:-/scratch/$USER/inferencex-dsv4}"
export RUBY_AIPERF_CACHE="${RUBY_AIPERF_CACHE:-/scratch/$USER/inferencex/aiperf-cache}"
export BENCHMARK_GPU_LABEL="${BENCHMARK_GPU_LABEL:-mi355x}"

# Ruby allocates the full node. Match the official GPU_COUNT=TP behavior so
# framework workers cannot accidentally use GPUs outside this recipe arm.
if [[ -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    visible_devices=""
    for ((gpu = 0; gpu < TP; gpu++)); do
        [[ -n "$visible_devices" ]] && visible_devices+=","
        visible_devices+="$gpu"
    done
    export ROCR_VISIBLE_DEVICES="$visible_devices"
fi
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ROCR_VISIBLE_DEVICES}"

case "${RUBY_DSV4_SMOKE:-0}" in
    1|true|TRUE|yes|YES)
        export DURATION="${DURATION:-60}"
        export AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-1}"
        export AIPERF_EXPERIMENTAL_FAST=0
        export AIPERF_UNSAFE_OVERRIDE=true
        ;;
    *)
        export DURATION="${DURATION:-3600}"
        ;;
esac

exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_mi350x-ruby.sh"
