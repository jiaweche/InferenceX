#!/usr/bin/env bash
set -euo pipefail

# Run a single-node InferenceX recipe from an already allocated Ruby MI350X
# worker. Ruby exposes the GPUs through Docker rather than Slurm Pyxis/enroot.

node="$(hostname -s)"
if [[ "$node" != cv350-rck-* ]]; then
    echo "Error: this launcher must run on a Ruby compute node (cv350-rck-*), got $node" >&2
    exit 1
fi

required_vars=(
    IMAGE MODEL MODEL_PREFIX PRECISION FRAMEWORK EXP_NAME
    TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB DURATION
)
missing_vars=()
for var_name in "${required_vars[@]}"; do
    if [[ -z "${!var_name:-}" ]]; then
        missing_vars+=("$var_name")
    fi
done
if (( ${#missing_vars[@]} )); then
    printf 'Error: missing required environment variable: %s\n' "${missing_vars[@]}" >&2
    exit 1
fi

if command -v squeue >/dev/null 2>&1; then
    allocation_id="$(
        squeue --noheader --user "$USER" --nodelist "$node" --states RUNNING --format '%A' |
            awk 'NF { print; exit }'
    )"
    if [[ -z "$allocation_id" ]]; then
        echo "Error: $USER has no running Slurm allocation on $node" >&2
        exit 1
    fi
fi

export PP_SIZE="${PP_SIZE:-1}"
export DCP_SIZE="${DCP_SIZE:-1}"
export PCP_SIZE="${PCP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-1}"
export DP_ATTENTION="${DP_ATTENTION:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"
export SPEC_DECODING="${SPEC_DECODING:-none}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
export PORT="${PORT:-8888}"
export SCENARIO_TYPE="${SCENARIO_TYPE:-agentic-coding}"
export SCENARIO_SUBDIR="${SCENARIO_SUBDIR:-agentic/}"
export IS_AGENTIC="${IS_AGENTIC:-1}"
export RUNNER_TYPE="${RUNNER_TYPE:-cluster:ruby-mi350x}"

repo_root="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
if [[ "$SCENARIO_TYPE" == "agentic-coding" &&
      ! -f "$repo_root/utils/aiperf/pyproject.toml" ]]; then
    echo "Error: the pinned AIPerf submodule is missing." >&2
    echo "Run: git -C '$repo_root' submodule update --init --recursive" >&2
    exit 1
fi

scratch_root="${RUBY_SCRATCH_ROOT:-/scratch/$USER/inferencex}"
model_root="${RUBY_MODEL_ROOT:-$scratch_root/models}"
model_slug="${MODEL##*/}"
host_model_path="$model_root/$model_slug"
host_hf_cache="${RUBY_HF_CACHE:-$scratch_root/hf-cache}"
host_hf_home="${RUBY_HF_HOME:-$scratch_root/hf-home}"
host_aiperf_cache="${RUBY_AIPERF_CACHE:-$scratch_root/aiperf-cache}"
host_runtime="${RUBY_RUNTIME_DIR:-$scratch_root/runtime}"

run_tag="${RUBY_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULT_FILENAME="${RESULT_FILENAME:-${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP_SIZE}_conc${CONC}_${run_tag}}"
host_result_dir="${RUBY_RESULT_DIR:-$scratch_root/runs/$RESULT_FILENAME}"

mkdir -p \
    "$host_model_path" \
    "$host_hf_cache" \
    "$host_hf_home" \
    "$host_aiperf_cache" \
    "$host_runtime/tmp" \
    "$host_result_dir"

export MODEL_PATH="/models/$model_slug"
export HF_HOME="/hf_home"
export HF_HUB_CACHE="/hf_hub_cache"
export AIPERF_DATASET_MMAP_CACHE_DIR="/aiperf_mmap_cache"
export AIPERF_RUNTIME_DIR="/runtime/aiperf-$RESULT_FILENAME"
export TMPDIR="/runtime/tmp"
export XDG_CACHE_HOME="/runtime/cache"
export RESULT_DIR="/results"
export AGENTIC_OUTPUT_DIR="/results"
export INFMAX_CONTAINER_WORKSPACE="/workspace"

spec_suffix=""
if [[ "$SPEC_DECODING" == "mtp" ]]; then
    spec_suffix="_mtp"
fi
gpu_recipe_label="${BENCHMARK_GPU_LABEL:-mi355x}"
script_base="${EXP_NAME%%_*}_${PRECISION}_${gpu_recipe_label}"
script_with_framework="benchmarks/single_node/${SCENARIO_SUBDIR}${script_base}_${FRAMEWORK}${spec_suffix}.sh"
script_fallback="benchmarks/single_node/${SCENARIO_SUBDIR}${script_base}${spec_suffix}.sh"
if [[ -f "$repo_root/$script_with_framework" ]]; then
    benchmark_script="$script_with_framework"
elif [[ -f "$repo_root/$script_fallback" ]]; then
    benchmark_script="$script_fallback"
else
    echo "Error: benchmark script not found: $script_with_framework or $script_fallback" >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker pull "$IMAGE"
fi

video_gid="$(getent group video | cut -d: -f3)"
render_gid="$(getent group render | cut -d: -f3)"
container_name="inferencex-${USER}-${run_tag,,}"

{
    printf 'started_at=%s\n' "$(date -u +%FT%TZ)"
    printf 'node=%s\n' "$node"
    printf 'slurm_job_id=%s\n' "${allocation_id:-${SLURM_JOB_ID:-}}"
    printf 'repository_sha=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'image=%s\n' "$IMAGE"
    printf 'image_id=%s\n' "$(docker image inspect "$IMAGE" --format '{{.Id}}')"
    printf 'model=%s\n' "$MODEL"
    printf 'host_model_path=%s\n' "$host_model_path"
    printf 'benchmark_script=%s\n' "$benchmark_script"
    printf 'result_filename=%s\n' "$RESULT_FILENAME"
} > "$host_result_dir/launch_metadata.txt"

cleanup() {
    rc=$?
    trap - EXIT INT TERM HUP
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    sudo -n chown -R "$(id -u):$(id -g)" "$host_result_dir" "$host_runtime" 2>/dev/null || true
    printf 'Ruby result directory: %s\n' "$host_result_dir"
    exit "$rc"
}
trap cleanup EXIT INT TERM HUP

env_names=(
    MODEL MODEL_PREFIX MODEL_PATH IMAGE PRECISION FRAMEWORK EXP_NAME
    TP PP_SIZE DCP_SIZE PCP_SIZE EP_SIZE DP_SIZE DP_ATTENTION CONC
    KV_OFFLOADING KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA
    TOTAL_CPU_DRAM_GB DURATION SPEC_DECODING
    EVAL_ONLY RUN_EVAL RANDOM_RANGE_RATIO PORT
    SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC RUNNER_TYPE
    RESULT_FILENAME RESULT_DIR AGENTIC_OUTPUT_DIR INFMAX_CONTAINER_WORKSPACE
    HF_TOKEN HF_HOME HF_HUB_CACHE HF_XET_HIGH_PERFORMANCE
    AIPERF_DATASET_MMAP_CACHE_DIR AIPERF_RUNTIME_DIR
    AIPERF_WARMUP_REQUESTS_PER_LANE AIPERF_EXPERIMENTAL_FAST
    AIPERF_UNSAFE_OVERRIDE AIPERF_FAILED_REQUEST_THRESHOLD
    AIPERF_LIVE_FAILED_REQUEST_THRESHOLD ENABLE_AGENTX_POWER
    HICACHE_RATIO HICACHE_WRITE_POLICY HICACHE_IO_BACKEND HICACHE_MEM_LAYOUT
    ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    TMPDIR XDG_CACHE_HOME
)
docker_env=()
for var_name in "${env_names[@]}"; do
    if [[ -n "${!var_name+x}" ]]; then
        docker_env+=(--env "$var_name")
    fi
done

set -x
docker run --rm \
    --name "$container_name" \
    --network=host \
    --ipc=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add "$video_gid" \
    --group-add "$render_gid" \
    --shm-size=128g \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864 \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --volume "$repo_root:/workspace" \
    --volume "$model_root:/models" \
    --volume "$host_hf_cache:/hf_hub_cache" \
    --volume "$host_hf_home:/hf_home" \
    --volume "$host_aiperf_cache:/aiperf_mmap_cache" \
    --volume "$host_runtime:/runtime" \
    --volume "$host_result_dir:/results" \
    --workdir=/workspace \
    "${docker_env[@]}" \
    --entrypoint=/bin/bash \
    "$IMAGE" \
    "$benchmark_script"
