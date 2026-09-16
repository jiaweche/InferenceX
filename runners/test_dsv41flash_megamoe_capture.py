import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_shape_capture_forces_eager_and_opens_gate_at_profiling(
    tmp_path: Path,
) -> None:
    active_file = tmp_path / "shape.active"
    capture_file = tmp_path / "shape.jsonl"
    env = {
        **os.environ,
        "MODEL": "fixture",
        "TP": "4",
        "CONC": "8",
        "KV_OFFLOADING": "none",
        "TOTAL_CPU_DRAM_GB": "0",
        "DURATION": "60",
        "RESULT_DIR": str(tmp_path),
        "EVAL_ONLY": "false",
        "PORT": "18888",
        "VLLM_MOE_SHAPE_CAPTURE": "1",
        "VLLM_MOE_SHAPE_CAPTURE_PATH": str(capture_file),
        "VLLM_MOE_SHAPE_CAPTURE_ACTIVE_FILE": str(active_file),
        "VLLM_AITER_MEGA_MOE_V2": "1",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.8",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
source() { :; }
check_env_vars() { :; }
require_agentic_kv_offload_none() { :; }
hf() { :; }
resolve_trace_source() { :; }
install_agentic_deps() { :; }
wait_for_server_ready() { :; }
build_replay_cmd() { :; }
run_eval() { :; }
run_agentic_replay_and_write_outputs() {
    printf '%s\n' 'Phase profiling (profiling) started' > "$RESULT_DIR/benchmark.log"
    for ((attempt = 0; attempt < 200; attempt++)); do
        if [[ -f "$VLLM_MOE_SHAPE_CAPTURE_ACTIVE_FILE" ]]; then
            printf '%s\n' "$VLLM_MOE_SHAPE_CAPTURE_PATH" > "$RESULT_DIR/capture_seen.txt"
            return 0
        fi
        sleep 0.01
    done
    return 1
}
stop_background_process_tree() {
    : > "$RESULT_DIR/stop_server"
    wait "$1"
}
vllm() {
    command python3 -c 'import json,os,sys; json.dump(sys.argv[1:],open(os.environ["RESULT_DIR"]+"/args.json","w"))' "$@"
    while [[ ! -f "$RESULT_DIR/stop_server" ]]; do sleep 0.01; done
}
builtin source "$1/benchmarks/single_node/agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh"
""",
            "bash",
            str(ROOT),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "args.json").read_text())
    assert "--enforce-eager" in args
    assert "--enable-expert-parallel" in args
    assert args[args.index("--all2all-backend") + 1] == "mori_high_throughput"
    assert args[args.index("--max-cudagraph-capture-size") + 1] == "128"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.8"
    kernel_config = json.loads(args[args.index("--kernel-config") + 1])
    assert kernel_config == {
        "enable_aiter_mega_moe_v2": True,
        "aiter_mega_moe_v2_max_tokens": 8192,
        "aiter_mega_moe_v2_token_allowlist": [
            888,
            889,
            2625,
            2626,
            7100,
            7101,
        ],
    }
    assert (tmp_path / "capture_seen.txt").read_text().strip() == str(capture_file)
    assert not active_file.exists()
