#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPUs assigned to this run}"
BRANCH_EMBEDDING_GPUS="${EL_TSBPO_BRANCH_EMBEDDING_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a embedding_gpu_list <<<"$BRANCH_EMBEDDING_GPUS"
(( ${#embedding_gpu_list[@]} > 0 )) || {
    printf 'Branch encoder GPU list must not be empty.\n' >&2
    exit 2
}
for embedding_gpu in "${embedding_gpu_list[@]}"; do
    if [[ -z "$embedding_gpu" || ",$CUDA_VISIBLE_DEVICES," != *",$embedding_gpu,"* ]]; then
        printf 'Branch encoder GPU %s is not assigned to this run (%s).\n' \
            "$embedding_gpu" "$CUDA_VISIBLE_DEVICES" >&2
        exit 2
    fi
done
export EL_TSBPO_BRANCH_EMBEDDING_CUDA_VISIBLE_DEVICES="$BRANCH_EMBEDDING_GPUS"

WORK_DIR="${EL_TSBPO_WORK_DIR:-$ROOT/.local}"
LOG_DIR="${EL_TSBPO_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/el-tsbpo/logs}"
MODEL="${EL_TSBPO_MODEL:-$WORK_DIR/models/alfworld-sft}"
EMBEDDING="${EL_TSBPO_EMBEDDING_MODEL:-$WORK_DIR/models/Qwen3-Embedding-0.6B}"
DATA="$WORK_DIR/data/alfworld"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${XDG_CACHE_HOME:-$HOME/.cache}/alfworld}"
export TMPDIR="${TMPDIR:-/tmp/el-tsbpo-${UID}}"
export RAY_DISABLE_DASHBOARD=1 RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false
export MULTIPROCESSING_START_METHOD=spawn
export OMP_NUM_THREADS=1
[[ -f "$MODEL/config.json" && -d "$EMBEDDING" ]] || {
    printf 'Set EL_TSBPO_MODEL and EL_TSBPO_EMBEDDING_MODEL to local model directories.\n' >&2
    exit 1
}
[[ -d "$ALFWORLD_DATA/json_2.1.1" && -d "$ALFWORLD_DATA/logic" ]] || {
    printf 'ALFWorld data is missing; run alfworld-download -f or set ALFWORLD_DATA.\n' >&2
    exit 1
}
mkdir -p "$DATA" "$TMPDIR" "$WORK_DIR/cache"
GPU_COUNT="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
(( 90 % GPU_COUNT == 0 && 240 % GPU_COUNT == 0 )) || {
    printf 'ALFWorld needs a GPU count dividing both the 90 root rollouts and 240-turn PPO minibatch.\n' >&2
    exit 1
}

python3 -m examples.data_preprocess.prepare_simple \
    --mode text --local_dir "$DATA" --train_data_size 15 --val_data_size 120

exec python3 -m verl.trainer.main_ppo --config-name el_tsbpo \
    hydra.run.dir="$LOG_DIR/alfworld/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
    data.train_files="$DATA/text/train.parquet" \
    data.val_files="$DATA/text/test.parquet" \
    data.train_batch_size=15 data.val_batch_size=120 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enforce_eager=false \
    +actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,8,16,24,32,48,64]' \
    algorithm.el_tsbpo.root_budget=90 \
    algorithm.el_tsbpo.branch_budget=120 \
    algorithm.el_tsbpo.graph_gigpo_segment_balance=false \
    algorithm.el_tsbpo.agent_dps_root_embedding_model_path="$EMBEDDING" \
    algorithm.el_tsbpo.agent_dps_root_embedding_cache_dir="$WORK_DIR/cache/alfworld_tasks" \
    algorithm.el_tsbpo.agent_dps_branch_similarity.cache_dir="$WORK_DIR/cache/alfworld_states" \
    algorithm.el_tsbpo.agent_dps_branch_similarity.timeout_seconds="${EL_TSBPO_BRANCH_EMBEDDING_TIMEOUT_SECONDS:-180}" \
    env.env_name=alfworld/AlfredTWEnv \
    env.max_steps=50 \
    trainer.experiment_name=alfworld \
    trainer.n_gpus_per_node="$GPU_COUNT" \
    "$@"
