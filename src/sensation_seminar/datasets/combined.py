"""Minimal deterministic concatenation for real and synthetic datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class CombinedDataset:
    def __init__(self, *datasets: Sequence[Any]):
        self.datasets = tuple(dataset for dataset in datasets if len(dataset))
        self.lengths = tuple(len(dataset) for dataset in self.datasets)

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for dataset, length in zip(self.datasets, self.lengths):
            if index < length:
                return dataset[index]
            index -= length
        raise IndexError(index)
