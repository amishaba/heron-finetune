"""Download LEDGAR (LexGLUE) into data/ledgar/ as raw CSVs: train.csv, validation.csv, test.csv with
columns `text` (a contract provision) and `label` (its clause type, one of 100). No task.yaml: this folder
is for the planning agent (`ft run data/ledgar`).

Streams the ~28 MB of parquet with the Hugging Face cache pointed at a temp folder, so nothing is left
in ~/.cache/huggingface. Run: .venv/bin/python examples/ledgar/fetch.py
"""

import csv
import os
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "ledgar"

with tempfile.TemporaryDirectory() as tmp:
    os.environ["HF_HOME"] = tmp
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        ds = load_dataset("coastalcph/lex_glue", "ledgar", split=split, streaming=True)
        names = ds.features["label"].names
        n = 0
        with (OUT / f"{split}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            for row in ds:
                w.writerow([row["text"], names[row["label"]]])
                n += 1
        print(f"{split}: {n} rows")
