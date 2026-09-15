#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4.1-Flash AgentX preset for an already allocated Ruby MI350X node.
# Keep this separate from the DeepSeek-V4-Pro SGLang preset: V4.1 uses the
# validated vLLM/DSpark/AITER recipe and an independently pinned checkpoint.

export IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657}"
export MODEL="${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
export MODEL_REVISION="${MODEL_REVISION:-dba1be0a40aa45a94ad051997016db3960a90277}"
export MODEL_PREFIX="${MODEL_PREFIX:-dsv41flash}"
export PRECISION="${PRECISION:-fp4}"
export FRAMEWORK="${FRAMEWORK:-vllm}"
export SPEC_DECODING="${SPEC_DECODING:-mtp}"

export TP="${TP:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export DCP_SIZE="${DCP_SIZE:-1}"
export PCP_SIZE="${PCP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-1}"
export DP_ATTENTION="${DP_ATTENTION:-false}"
export CONC="${CONC:-1}"
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-0}"

export EXP_NAME="${EXP_NAME:-dsv41flash_tp${TP}_conc${CONC}_kvnone_dspark}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"
export ENABLE_AGENTX_POWER="${ENABLE_AGENTX_POWER:-0}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126}"
export RUBY_SCRATCH_ROOT="${RUBY_SCRATCH_ROOT:-/scratch/$USER/inferencex-dsv41flash}"
export RUBY_AIPERF_CACHE="${RUBY_AIPERF_CACHE:-/scratch/$USER/inferencex/aiperf-cache}"
export BENCHMARK_GPU_LABEL="${BENCHMARK_GPU_LABEL:-mi355x}"

if [[ -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    visible_devices=""
    for ((gpu = 0; gpu < TP; gpu++)); do
        [[ -n "$visible_devices" ]] && visible_devices+=","
        visible_devices+="$gpu"
    done
    export ROCR_VISIBLE_DEVICES="$visible_devices"
fi

case "${RUBY_DSV41FLASH_SMOKE:-0}" in
    1|true|TRUE|yes|YES)
        export DURATION="${DURATION:-60}"
        export AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-1}"
        export AIPERF_EXPERIMENTAL_FAST=0
        export AIPERF_UNSAFE_OVERRIDE=true
        ;;
    *)
        export DURATION="${DURATION:-1200}"
        export AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-1}"
        export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-1}"
        ;;
esac

exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_mi350x-ruby.sh"
