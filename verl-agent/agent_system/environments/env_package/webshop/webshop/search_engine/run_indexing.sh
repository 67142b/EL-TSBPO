#!/usr/bin/env bash

set -euo pipefail

resources=(resources_100 resources resources_1k resources_100k)
indexes=(indexes_100 indexes indexes_1k indexes_100k)
threads="${WEBSHOP_INDEX_THREADS:-1}"
backup_dir="$(mktemp -d /tmp/webshop-index-backup.XXXXXX)"
build_succeeded=0

restore_or_cleanup() {
  local status=$?
  trap - EXIT

  if [[ "${build_succeeded}" -eq 0 ]]; then
    echo "Indexing failed; restoring the previous indexes." >&2
    for index in "${indexes[@]}"; do
      rm -rf -- "${index}"
      if [[ -e "${backup_dir}/${index}" ]]; then
        mv -- "${backup_dir}/${index}" "${index}"
      fi
    done
  fi
  rm -rf -- "${backup_dir}"
  exit "${status}"
}
trap restore_or_cleanup EXIT

# This Pyserini release has no --overwrite flag. Move the previous indexes
# aside so a failed rebuild can restore them intact.
for index in "${indexes[@]}"; do
  if [[ -e "${index}" ]]; then
    mv -- "${index}" "${backup_dir}/${index}"
  fi
done

for i in "${!indexes[@]}"; do
  python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "${resources[$i]}" \
    --index "${indexes[$i]}" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${threads}" \
    --storePositions --storeDocvectors --storeRaw
done

build_succeeded=1
