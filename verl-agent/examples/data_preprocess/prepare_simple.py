"""
Simplified data preparation script (no HuggingFace download needed).

For environments like ALFWorld/WebShop/Sokoban, the parquet data only indicates
the modality ("text" or "visual"). The actual task content comes from the
environment itself. This script generates parquet files with the correct schema
and row count without downloading any external dataset.

Output: ~/data/verl-agent/{mode}/train.parquet, test.parquet
"""

import os
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def build_rows(split: str, n: int, mode: str):
    rows = []
    for i in range(n):
        rows.append({
            "data_source": mode,
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": i},
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="text", choices=["visual", "text"])
    parser.add_argument("--local_dir", default="~/data/verl-agent/")
    parser.add_argument("--train_data_size", default=256, type=int)
    parser.add_argument("--val_data_size", default=256, type=int)
    args = parser.parse_args()

    print(f"processing data for mode: {args.mode}")
    args.local_dir = os.path.expanduser(args.local_dir)
    out_dir = os.path.join(args.local_dir, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    train_rows = build_rows("train", args.train_data_size, args.mode)
    test_rows = build_rows("test", args.val_data_size, args.mode)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"saved train ({len(train_rows)} rows) -> {train_path}")
    print(f"saved test  ({len(test_rows)} rows) -> {test_path}")


if __name__ == "__main__":
    main()
