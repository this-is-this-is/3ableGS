"""Print or save a resolved config without loading data or starting training."""

import argparse
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import load_training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("opts", nargs="*", help="key=value overrides")
    args = parser.parse_args()
    config = load_training_config(args.config_file, args.opts)
    resolved = OmegaConf.to_yaml(config, resolve=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(resolved, encoding="utf-8")
        print(args.out)
    else:
        print(resolved, end="")


if __name__ == "__main__":
    main()
