#!/usr/bin/env python3
"""Backward-compatible launcher alias for Base v5."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_base_v5 import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
