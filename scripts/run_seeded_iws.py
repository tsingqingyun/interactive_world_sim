#!/usr/bin/env python3
"""Run the unchanged IWS entrypoint after fixing all process-level RNG seeds."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("entrypoint", type=Path)
    parser.add_argument("hydra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.argv = [str(args.entrypoint), *args.hydra_args]
    runpy.run_path(str(args.entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
