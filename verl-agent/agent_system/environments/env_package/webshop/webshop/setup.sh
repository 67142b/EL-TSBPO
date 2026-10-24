#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 [-d small|all]"
  echo "  -d small|all  Download the 1,000-product subset or the full catalog"
  exit 1
}

data_size=""
while getopts "d:" flag; do
  case "${flag}" in
    d) data_size="${OPTARG}" ;;
    *) usage ;;
  esac
done

if [[ "${data_size}" != "small" && "${data_size}" != "all" ]]; then
  usage
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SEARCH_DIR="${SCRIPT_DIR}/search_engine"
HF_DATA_REPO="${WEBSHOP_HF_DATA_REPO:-YWZBrandon/webshop-data}"

python - <<'PY'
import sys

if not ((3, 10, 1) <= sys.version_info[:3] < (3, 11)):
    raise SystemExit(
        "WebShop integration requires Python >=3.10.1,<3.11. "
        "Create the environment with python=3.10.19."
    )
PY

mkdir -p "${DATA_DIR}"

echo "Installing WebShop dependencies..."
conda_pkg_installed() {
  local package_name="$1"
  conda list --name "${CONDA_DEFAULT_ENV:-}" "${package_name}" 2>/dev/null \
    | awk -v package="${package_name}" '$1 == package { found=1 } END { exit(found ? 0 : 1) }'
}

if ! conda_pkg_installed mkl; then
  conda install -y mkl
fi
if ! python -c 'import faiss' >/dev/null 2>&1; then
  conda install -y -c conda-forge faiss-cpu
fi
if ! command -v java >/dev/null 2>&1; then
  conda install -y -c conda-forge openjdk=11
fi
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

download_data_file() {
  local filename="$1"
  local drive_id="$2"
  local target="${DATA_DIR}/${filename}"
  local partial="${target}.part.$$"
  local hf_url="https://huggingface.co/datasets/${HF_DATA_REPO}/resolve/main/${filename}?download=true"

  if [[ -s "${target}" ]]; then
    echo "Using existing ${filename}"
    return 0
  fi

  rm -f "${partial}"
  if command -v gdown >/dev/null 2>&1 && \
      gdown --id "${drive_id}" -O "${partial}"; then
    if [[ -s "${partial}" ]]; then
      mv -f "${partial}" "${target}"
      return 0
    fi
  fi
  rm -f "${partial}"

  echo "Google Drive download failed for ${filename}; trying ${hf_url}"
  curl --fail --location --retry 3 --retry-delay 2 \
    --output "${partial}" "${hf_url}"
  test -s "${partial}"
  mv -f "${partial}" "${target}"
}

download_data_file items_shuffle_1000.json 1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
download_data_file items_ins_v2_1000.json 1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
if [[ "${data_size}" == "all" ]]; then
  download_data_file items_shuffle.json 1A2whVgOO0euk5O13n2iYDM0bQRkkRduB
  download_data_file items_ins_v2.json 1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi
fi
download_data_file items_human_ins.json 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O

install_spacy_model() {
  local package_name="$1"
  local wheel_url="$2"
  if python -c "import spacy; spacy.load('${package_name}')" >/dev/null 2>&1; then
    echo "Using existing ${package_name}"
    return 0
  fi
  python -m pip install "${wheel_url}"
}

# The spaCy CLI can construct a malformed GitHub URL in some mirrored setups.
# Install the pinned official release wheels directly instead.
install_spacy_model \
  en_core_web_sm \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
if [[ "${WEBSHOP_INSTALL_LG:-0}" == "1" ]]; then
  install_spacy_model \
    en_core_web_lg \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
fi

mkdir -p "${SEARCH_DIR}/resources" "${SEARCH_DIR}/resources_100" \
  "${SEARCH_DIR}/resources_1k" "${SEARCH_DIR}/resources_100k"
cd "${SEARCH_DIR}"

if [[ "${data_size}" == "all" ]]; then
  product_file="${DATA_DIR}/items_shuffle.json"
  attr_file="${DATA_DIR}/items_ins_v2.json"
  num_products=-1
else
  product_file="${DATA_DIR}/items_shuffle_1000.json"
  attr_file="${DATA_DIR}/items_ins_v2_1000.json"
  num_products=1000
fi

resource_file="${SEARCH_DIR}/resources/documents.jsonl"
resource_count=0
if [[ -f "${resource_file}" ]]; then
  resource_count="$(wc -l < "${resource_file}")"
fi

resources_current=0
if [[ "${resource_file}" -nt "${product_file}" && \
      "${resource_file}" -nt "${attr_file}" ]]; then
  if [[ "${data_size}" == "small" && "${resource_count}" -eq 1000 ]]; then
    resources_current=1
  elif [[ "${data_size}" == "all" && "${resource_count}" -gt 1000 ]]; then
    resources_current=1
  fi
fi

if [[ "${resources_current}" -eq 1 ]]; then
  echo "Search resources for ${data_size} data already exist; skipping conversion."
else
  python convert_product_file_format.py \
    --file-path "${product_file}" \
    --attr-path "${attr_file}" \
    --num-products "${num_products}"
  resource_count="$(wc -l < "${resource_file}")"
fi

indexes_match_resources() {
  python - <<'PY'
import json

from pyserini.index.lucene import IndexReader

resources = {
    "indexes": "resources/documents.jsonl",
    "indexes_100": "resources_100/documents.jsonl",
    "indexes_1k": "resources_1k/documents.jsonl",
    "indexes_100k": "resources_100k/documents.jsonl",
}

for index_path, resource_path in resources.items():
    with open(resource_path, encoding="utf-8") as resource_file:
        expected_documents = sum(
            bool(json.loads(line).get("contents", "").strip())
            for line in resource_file
        )
    if IndexReader(index_path).stats()["documents"] != expected_documents:
        raise SystemExit(1)
PY
}

index_marker="${SEARCH_DIR}/.index_mode"
existing_index_mode=""
if [[ -f "${index_marker}" ]]; then
  existing_index_mode="$(<"${index_marker}")"
fi
if [[ "${existing_index_mode}" != "${data_size}" ]] || \
    ! indexes_match_resources >/dev/null 2>&1; then
  bash run_indexing.sh
  indexes_match_resources
  printf '%s\n' "${data_size}" > "${index_marker}"
else
  echo "Lucene indexes for ${data_size} data already exist; skipping rebuild."
fi

echo "WebShop setup complete (${data_size} data)."
