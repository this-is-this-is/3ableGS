"""Resolve repository YAML inheritance without importing training dependencies."""

from pathlib import Path
from typing import Iterable, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def load_config_with_base(path: Union[str, Path], _stack=()) -> DictConfig:
    """Merge relative base files left-to-right, then apply the child config.

    ``base_config`` can be one path or a list of paths, each relative to the
    YAML declaring it. Lists in configuration values replace inherited lists.
    Only the current ancestry is checked for cycles, so shared bases are valid.
    """
    path = Path(path).resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Circular base_config inheritance: {chain}")
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected a YAML mapping in {path}")
    bases = config.pop("base_config", None)
    if bases is None:
        return config
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, (list, ListConfig)) or not all(
        isinstance(base, str) and base for base in bases
    ):
        raise ValueError(f"base_config must be a path or list of paths in {path}")
    inherited = [
        load_config_with_base(path.parent / base, (*_stack, path))
        for base in bases
    ]
    return OmegaConf.merge(OmegaConf.create({}), *inherited, config)


def load_training_config(
    path: Union[str, Path],
    overrides: Iterable[str] = (),
    *,
    config_root: Union[str, Path] = CONFIG_ROOT,
) -> DictConfig:
    """Resolve dataset defaults < inherited YAML < explicit CLI overrides.

    Dataset aliases follow the existing ``dataset: waymo/1cams`` convention.
    A CLI dataset alias selects the defaults before data overrides are merged.
    Data-only configs are also supported for preparation and inspection.
    """
    config = load_config_with_base(path)
    cli = OmegaConf.from_dotlist(list(overrides or ()))
    dataset = cli.pop("dataset", config.pop("dataset", None))
    if dataset is None:
        if "data" not in config and "data" not in cli:
            raise ValueError("Please specify dataset in config or data in config")
        defaults = OmegaConf.create({})
    else:
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("dataset must be a nonempty dataset configuration name")
        defaults = load_config_with_base(Path(config_root) / "datasets" / f"{dataset}.yaml")
    return OmegaConf.merge(defaults, config, cli)
