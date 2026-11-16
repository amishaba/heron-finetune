#!/usr/bin/env bash
# Downloads the BANKING77 CSVs (PolyAI, CC-BY-4.0) into data/banking77/.
set -euo pipefail
cd "$(dirname "$0")/../.."
dest=data/banking77
base=https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data
mkdir -p "$dest"
for f in train.csv test.csv; do
  curl -fsSL "$base/$f" -o "$dest/$f"
  echo "$dest/$f: $(($(wc -l < "$dest/$f") - 1)) rows"
done
