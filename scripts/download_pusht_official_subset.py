#!/usr/bin/env python3
"""Download a deterministic official MuJoCo PushT subset as Dataset A."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "yixuan1999/interactive-world-sim-mujoco-data"
OFFICIAL_TRAIN_EPISODES = 10000
OFFICIAL_VAL_EPISODES = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    if args.train_episodes + args.test_episodes > OFFICIAL_TRAIN_EPISODES:
        raise ValueError("Requested train+test exceeds the official train split")
    if args.val_episodes > OFFICIAL_VAL_EPISODES:
        raise ValueError("Requested validation count exceeds the official val split")

    rng = np.random.default_rng(args.seed)
    shuffled_train = rng.permutation(OFFICIAL_TRAIN_EPISODES)
    train_ids = sorted(shuffled_train[: args.train_episodes].tolist())
    test_ids = sorted(
        shuffled_train[
            args.train_episodes : args.train_episodes + args.test_episodes
        ].tolist()
    )
    val_ids = list(range(args.val_episodes))

    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (args.output_root / split).mkdir(parents=True, exist_ok=True)

    split_manifest = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "seed": args.seed,
        "selection": (
            "random without replacement from official train; official val prefix"
        ),
        "train_source_ids": train_ids,
        "val_source_ids": val_ids,
        "test_source_ids": test_ids,
    }
    manifest_path = args.output_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2) + "\n")

    def download(source_split: str, episode_id: int, output_split: str) -> None:
        destination = args.output_root / output_split / f"episode_{episode_id}.hdf5"
        if destination.is_file() and destination.stat().st_size > 0:
            return
        filename = f"{source_split}/episode_{episode_id}.hdf5"
        for attempt in range(1, args.max_retries + 1):
            try:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=REPO_ID,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=args.output_root,
                    )
                )
                if downloaded.resolve() != destination.resolve():
                    shutil.move(str(downloaded), destination)
                return
            except Exception:
                if attempt == args.max_retries:
                    raise
                delay = min(2 ** (attempt - 1), 30)
                print(
                    f"Retrying {filename} in {delay}s "
                    f"(attempt {attempt + 1}/{args.max_retries})",
                    flush=True,
                )
                time.sleep(delay)

    for episode_id in train_ids:
        download("train", episode_id, "train")
    for episode_id in test_ids:
        download("train", episode_id, "test")
    for episode_id in val_ids:
        download("val", episode_id, "val")

    print(
        json.dumps(
            {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
