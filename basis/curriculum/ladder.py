"""
Curriculum ladder scaffolding for Reflexive Formal Learning alignment.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from basis.core import CurriculumIndex, CurriculumTier


class CurriculumLadder:
    """Immutable container for curriculum tiers."""

    def __init__(self, tiers: Iterable[CurriculumTier]):
        ordered = sorted(tiers, key=lambda tier: tier.identifier)
        self._tiers = tuple(ordered)
        self._index: CurriculumIndex = {tier.identifier: tier for tier in ordered}

    def __iter__(self) -> Iterator[CurriculumTier]:
        return iter(self._tiers)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._tiers)

    def tier(self, identifier: str) -> CurriculumTier:
        return self._index[identifier]

    def to_index(self) -> CurriculumIndex:
        return dict(self._index)

    def to_json_ready(self) -> Sequence[Mapping[str, object]]:
        return [asdict(tier) for tier in self._tiers]


def ladder_from_dict(rows: Sequence[Mapping[str, object]]) -> CurriculumLadder:
    """Create a ladder from plain dictionaries."""
    tiers = []
    for row in rows:
        tier = CurriculumTier(
            identifier=str(row["identifier"]),
            title=str(row["title"]),
            description=str(row.get("description", "")),
            prerequisites=tuple(row.get("prerequisites", [])),
            objectives=tuple(row.get("objectives", [])),
        )
        tiers.append(tier)
    return CurriculumLadder(tiers)


def ladder_from_json(path: Path) -> CurriculumLadder:
    """Load a ladder definition from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Curriculum ladder JSON must be a list of tier objects.")
    return ladder_from_dict(data)


def ladder_to_json(ladder: CurriculumLadder, path: Path) -> None:
    """Persist the ladder in a deterministic JSON format."""
    payload = ladder.to_json_ready()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

