from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """
    SafeLoader that rejects duplicate mapping keys instead of keeping the last.
    """


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    """
    Build a YAML mapping, rejecting duplicate keys.

    :param loader: Active loader.
    :param node: Mapping node.
    :param deep: Construct nested objects eagerly.
    :return: The mapping.
    :raises ValueError: On a duplicate key.
    """
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}: "
                "the later definition would silently win"
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass
class DatasetSpec:
    """
    A reference dataset: song dirs with ``lyrics.json``.
    """

    dir: Path
    expected: int | None = None


@dataclass
class EnsembleSpec:
    """
    Loaded ensemble registry.
    """

    datasets: dict[str, DatasetSpec] = field(default_factory=dict)
    systems: dict[str, dict[str, Path]] = field(default_factory=dict)
    subsets: dict[str, list[str]] = field(default_factory=dict)

    def job_dir(self, system: str, dataset: str) -> Path:
        """
        Job dir holding *system*'s hypotheses for *dataset*.

        :param system: System name.
        :param dataset: Dataset name.
        :return: Job directory.
        :raises KeyError: If the system has no run registered for the dataset.
        """
        if system not in self.systems:
            raise KeyError(f"unknown system {system!r}")
        if dataset not in self.systems[system]:
            raise KeyError(f"no hypotheses registered for {system!r} on {dataset!r}")
        return self.systems[system][dataset]

    def members(self, subset: str) -> list[str]:
        """
        Ordered member list of a named subset, or an ad-hoc ``a,b,c`` list.

        :param subset: Subset name, or comma-separated system names.
        :return: Member names in alignment order.
        """
        if subset in self.subsets:
            return list(self.subsets[subset])
        if "," in subset:
            members = [m.strip() for m in subset.split(",") if m.strip()]
            unknown = [m for m in members if m not in self.systems]
            if unknown:
                raise KeyError(f"unknown systems in ad-hoc subset: {unknown}")
            return members
        raise KeyError(f"unknown subset {subset!r}")

    def add_system(self, arg: str, dataset: str) -> None:
        """
        Register an ad-hoc ``name=job_dir`` system for one dataset.

        :param arg: ``name=job_dir``.
        :param dataset: Dataset the job dir belongs to.
        """
        name, sep, path = arg.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--system must be name=job_dir, got {arg!r}")
        self.systems.setdefault(name, {})[dataset] = Path(path)

    def add_dataset(self, name: str, directory: Path) -> None:
        """
        Register (or override) a dataset directory.

        :param name: Dataset name.
        :param directory: Directory of song subdirectories.
        """
        self.datasets[name] = DatasetSpec(dir=Path(directory))


def load_spec(path: Path | None) -> EnsembleSpec:
    """
    Load an ensemble spec file; ``None`` returns an empty spec for ad-hoc use.

    ::

        root: /path/to/project   # relative paths resolve here; default: the spec's dir
        datasets:
          dev: data/dev
          test: {dir: data/test, expected: 100}   # expected: optional song-count check
        systems:
          whisper: {dev: jobs/whisper/dev, test: jobs/whisper/test}
          parakeet: {dev: jobs/parakeet/dev, test: jobs/parakeet/test}
        subsets:
          duo: [whisper, parakeet]   # first member is the alignment pivot and wins ties

    :param path: YAML/JSON spec path.
    :return: The loaded spec.
    :raises ValueError: On duplicate keys or subsets naming unknown systems.
    """
    spec = EnsembleSpec()
    if path is None:
        return spec
    with open(path) as f:
        raw = yaml.load(f, Loader=_UniqueKeyLoader) or {}
    root = Path(raw.get("root") or Path(path).parent)

    def resolve(p: str) -> Path:
        """
        Resolve a spec path against the spec root unless it is absolute.

        :param p: Path from the spec.
        :return: The resolved path.
        """
        q = Path(p)
        return q if q.is_absolute() else root / q

    for name, entry in (raw.get("datasets") or {}).items():
        if isinstance(entry, str):
            entry = {"dir": entry}
        spec.datasets[name] = DatasetSpec(dir=resolve(entry["dir"]), expected=entry.get("expected"))
    for name, runs in (raw.get("systems") or {}).items():
        spec.systems[name] = {ds: resolve(p) for ds, p in (runs or {}).items()}
    for name, members in (raw.get("subsets") or {}).items():
        unknown = [m for m in members if m not in spec.systems]
        if unknown:
            raise ValueError(f"subset {name!r} names unknown systems {unknown}")
        spec.subsets[name] = list(members)
    return spec
