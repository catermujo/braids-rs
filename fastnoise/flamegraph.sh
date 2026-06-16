#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example> [example args...]" >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"

example="$1"
shift

output_stem="$example"
for arg in "$@"; do
  safe_arg="${arg//[^A-Za-z0-9._-]/_}"
  output_stem+="_${safe_arg:-arg}"
done

output_dir="$script_dir/flamegraphs"
mkdir -p "$output_dir"

output_path="$output_dir/${output_stem}_flamegraph.svg"

echo "writing $output_path"

cd "$repo_dir"
CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph \
  -p braids-fastnoise \
  --example "$example" \
  --output "$output_path" \
  -- "$@"
