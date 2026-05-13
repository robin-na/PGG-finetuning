#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEW_ONLY="4bb49492edee4a8eb1758552a362a2cf"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  local expected_sha256="$3"

  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    local actual_sha256
    actual_sha256="$(shasum -a 256 "$out" | awk '{print $1}')"
    if [[ "$actual_sha256" == "$expected_sha256" ]]; then
      echo "OK existing: $out"
      return
    fi
    echo "Existing file has wrong/incomplete checksum, resuming: $out" >&2
  fi

  curl --fail --location --retry 5 --retry-delay 5 --continue-at - --output "$out" "$url"

  local actual_sha256
  actual_sha256="$(shasum -a 256 "$out" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Checksum mismatch for $out" >&2
    echo "expected: $expected_sha256" >&2
    echo "actual:   $actual_sha256" >&2
    exit 1
  fi
}

fetch_folder() {
  local folder_id="$1"
  local out_dir="$2"
  local include_external_large="${3:-0}"
  local url="https://api.osf.io/v2/nodes/3hvt2/files/osfstorage/${folder_id}/?view_only=${VIEW_ONLY}"
  local tmp_json
  tmp_json="$(mktemp)"

  mkdir -p "$out_dir"
  while [[ -n "$url" ]]; do
    curl --fail --location --silent --show-error --output "$tmp_json" "$url"
    jq -r '.data[] | select(.attributes.kind == "file") |
      [.links.download, .attributes.name, .attributes.size, .attributes.extra.hashes.sha256] | @tsv' "$tmp_json" |
      while IFS=$'\t' read -r download_url name size sha256; do
        if [[ "$include_external_large" != "1" && "$name" == "csv_pus.zip" ]]; then
          echo "Skipping large external file by default: $name ($size bytes)"
          continue
        fi
        echo "Downloading $name ($size bytes)"
        download_file "$download_url" "$out_dir/$name" "$sha256"
      done
    url="$(jq -r '.links.next // empty' "$tmp_json")"
  done
  rm -f "$tmp_json"
}

require_cmd curl
require_cmd jq
require_cmd shasum

mkdir -p "$ROOT_DIR/moral_machine/metadata"

curl --fail --location --silent --show-error \
  --output "$ROOT_DIR/moral_machine/metadata/osf_root_files.json" \
  "https://api.osf.io/v2/nodes/3hvt2/files/osfstorage/?view_only=${VIEW_ONLY}"

curl --fail --location --silent --show-error \
  --output "$ROOT_DIR/moral_machine/metadata/osf_datasets_files.json" \
  "https://api.osf.io/v2/nodes/3hvt2/files/osfstorage/5b53cdb369e43a000fd3a030/?view_only=${VIEW_ONLY}"

fetch_folder "5b53cdf8c86a8c001143da00" "$ROOT_DIR/moral_machine/raw/moral_machine_data"
fetch_folder "5bc8a900f97c5300161567bb" "$ROOT_DIR/moral_machine/raw/effect_sizes"
fetch_folder "5b53ce19c86a8c000f44a40f" "$ROOT_DIR/moral_machine/raw/external_data" "${DOWNLOAD_EXTERNAL_LARGE:-0}"

find "$ROOT_DIR/moral_machine/raw" -type f ! -name ".DS_Store" -print0 | sort -z | xargs -0 shasum -a 256 > "$ROOT_DIR/moral_machine/metadata/local_sha256.txt"
find "$ROOT_DIR/moral_machine/raw" -type f ! -name ".DS_Store" -print0 | sort -z | xargs -0 ls -lh > "$ROOT_DIR/moral_machine/metadata/local_files.txt"
