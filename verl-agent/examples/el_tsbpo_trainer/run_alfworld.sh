#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPUs assigned to this run}"

WORK_DIR="${EL_TSBPO_WORK_DIR:-$ROOT/.local}"
MODEL="${EL_TSBPO_MODEL:-$WORK_DIR/models/alfworld-sft}"
EMBEDDING="${EL_TSBPO_EMBEDDING_MODEL:-$WORK_DIR/models/Qwen3-Embedding-0.6B}"
DATA="$WORK_DIR/data/alfworld"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${XDG_CACHE_HOME:-$HOME/.cache}/alfworld}"
export TMPDIR="${TMPDIR:-$WORK_DIR/tmp}"
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

python3 -m examples.data_preprocess.prepare_simple \
    --mode text --local_dir "$DATA" --train_data_size 15 --val_data_size 120

exec python3 -m verl.trainer.main_ppo --config-name el_tsbpo \
    data.train_files="$DATA/text/train.parquet" \
    data.val_files="$DATA/text/test.parquet" \
    data.train_batch_size=15 data.val_batch_size=120 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enforce_eager=false \
    +actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,8,16,24,32,48,64]' \
    algorithm.el_tsbpo.root_budget=90 \
    algorithm.el_tsbpo.branch_budget=120 \
    algorithm.el_tsbpo.agent_dps_root_embedding_model_path="$EMBEDDING" \
    algorithm.el_tsbpo.agent_dps_root_embedding_cache_dir="$WORK_DIR/cache/alfworld_tasks" \
    algorithm.el_tsbpo.agent_dps_branch_similarity.cache_dir="$WORK_DIR/cache/alfworld_states" \
    env.env_name=alfworld/AlfredTWEnv \
    env.max_steps=50 \
    trainer.experiment_name=alfworld \
    trainer.n_gpus_per_node="$GPU_COUNT" \
    "$@"
