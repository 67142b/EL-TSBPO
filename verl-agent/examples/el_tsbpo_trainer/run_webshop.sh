#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPUs assigned to this run}"
export EL_TSBPO_BRANCH_EMBEDDING_CUDA_VISIBLE_DEVICES="${EL_TSBPO_BRANCH_EMBEDDING_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES%%,*}}"

WORK_DIR="${EL_TSBPO_WORK_DIR:-$ROOT/.local}"
LOG_DIR="${EL_TSBPO_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/el-tsbpo/logs}"
MODEL="${EL_TSBPO_MODEL:-$WORK_DIR/models/webshop-sft}"
EMBEDDING="${EL_TSBPO_EMBEDDING_MODEL:-$WORK_DIR/models/Qwen3-Embedding-0.6B}"
DATA="$WORK_DIR/data/webshop"
WEBSHOP="$ROOT/agent_system/environments/env_package/webshop/webshop"
export TMPDIR="${TMPDIR:-/tmp/el-tsbpo-${UID}}"
export RAY_DISABLE_DASHBOARD=1 RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false
export MULTIPROCESSING_START_METHOD=spawn
export OMP_NUM_THREADS=1
[[ -f "$MODEL/config.json" && -d "$EMBEDDING" ]] || {
    printf 'Set EL_TSBPO_MODEL and EL_TSBPO_EMBEDDING_MODEL to local model directories.\n' >&2
    exit 1
}
[[ -f "$WEBSHOP/data/items_shuffle_1000.json" && -d "$WEBSHOP/search_engine/indexes_1k" ]] || {
    printf 'WebShop assets are missing; run webshop/setup.sh -d small.\n' >&2
    exit 1
}
mkdir -p "$DATA" "$TMPDIR" "$WORK_DIR/cache"
GPU_COUNT="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
(( GPU_COUNT >= 2 && GPU_COUNT % 2 == 0 )) || {
    printf 'WebShop requires an even number of GPUs (tensor parallelism 2).\n' >&2
    exit 1
}

python3 -m examples.data_preprocess.prepare_simple \
    --mode text --local_dir "$DATA" --train_data_size 16 --val_data_size 128

exec python3 -m verl.trainer.main_ppo --config-name el_tsbpo \
    hydra.run.dir="$LOG_DIR/webshop/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
    data.train_files="$DATA/text/train.parquet" \
    data.val_files="$DATA/text/test.parquet" \
    data.train_batch_size=16 data.val_batch_size=128 data.max_prompt_length=4608 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=25600 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=25600 \
    algorithm.el_tsbpo.root_budget=96 \
    algorithm.el_tsbpo.branch_budget=128 \
    algorithm.el_tsbpo.agent_dps_root_embedding_model_path="$EMBEDDING" \
    algorithm.el_tsbpo.agent_dps_root_embedding_cache_dir="$WORK_DIR/cache/webshop_tasks" \
    algorithm.el_tsbpo.agent_dps_branch_similarity.cache_dir="$WORK_DIR/cache/webshop_states" \
    env.env_name=Webshop env.max_steps=15 \
    trainer.experiment_name=webshop \
    trainer.n_gpus_per_node="$GPU_COUNT" \
    trainer.val_before_train=true \
    "$@"
