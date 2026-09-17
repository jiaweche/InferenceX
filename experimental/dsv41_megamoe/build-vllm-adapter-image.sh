#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vllm_src="${VLLM_SRC:-/scratch/$USER/dsv41-megamoe/vllm-src}"
aiter_src="${AITER_SRC:-/scratch/$USER/dsv41-megamoe/aiter-vllm}"
mori_package="${MORI_PACKAGE:-/scratch/$USER/dsv41-megamoe/pydeps-vllm/mori}"
image_tag="${IMAGE_TAG:-dsv41-megamoe-adapter:vllm-e2944d9-aiter-22c8295-async-r1}"

expected_vllm="acb139b8725a46e1747c163f3d57089ff8bffee4"
expected_aiter="22c82955b41e2b482a99290537a044e6964499b1"
expected_mori_so="3df6da1342f1c9dc7923fd2620bb132b283b2063bf0040a88cb08056e136cfd5"

[[ "$(git -C "$vllm_src" rev-parse HEAD)" == "$expected_vllm" ]]
[[ "$(git -C "$aiter_src" rev-parse HEAD)" == "$expected_aiter" ]]
printf '%s  %s\n' "$expected_mori_so" "$mori_package/libmori_cco.so" |
    sha256sum --check --status

context="$(mktemp -d "${TMPDIR:-/scratch/$USER}/dsv41-adapter-image.XXXXXX")"
trap 'rm -rf -- "$context"' EXIT

vllm_files=(
    vllm/config/kernel.py
    vllm/distributed/device_communicators/all2all.py
    vllm/distributed/parallel_state.py
    vllm/forward_context.py
    vllm/model_executor/layers/fused_moe/config.py
    vllm/model_executor/layers/fused_moe/layer.py
    vllm/model_executor/layers/fused_moe/aiter_mega_moe_v2.py
    vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py
    vllm/model_executor/layers/quantization/mxfp4.py
    vllm/model_executor/model_loader/utils.py
)
for relative_path in "${vllm_files[@]}"; do
    install -D "$vllm_src/$relative_path" "$context/$relative_path"
done

aiter_kernel_dir="aiter/ops/flydsl/kernels"
for name in tensor_shim.py communication_ops_utils.py flydsl_dispatch_combine_intranode_op.py; do
    install -D \
        "$aiter_src/$aiter_kernel_dir/$name" \
        "$context/$aiter_kernel_dir/$name"
done
mkdir -p "$context/$aiter_kernel_dir/mega_moe"
cp -a "$aiter_src/$aiter_kernel_dir/mega_moe/." \
    "$context/$aiter_kernel_dir/mega_moe/"

mkdir -p "$context/mori"
cp -a "$mori_package/." "$context/mori/"
cp "$script_dir/Dockerfile.vllm-adapter" "$context/Dockerfile"

docker build --network=none --tag "$image_tag" "$context"
docker image inspect "$image_tag" --format '{{.Id}}'
