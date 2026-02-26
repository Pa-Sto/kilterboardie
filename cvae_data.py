import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class KilterSample:
    npy_path: str
    grade_v: int


class KilterRouteDataset(Dataset):
    """
    Dataset for Kilterboard route tensors.

    Each sample returns:
      - route: (4, H, W) float32 in {0,1}
      - static: (2, H, W) float32 (hold_presence, hold_size)
      - grade: int64, zero-based index (grade_v - grade_min)
    """

    def __init__(
        self,
        data_dir: str,
        grade_min: int = 3,
        grade_max: int = 13,
        include_unknown: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.grade_min = grade_min
        self.grade_max = grade_max
        self.include_unknown = include_unknown

        self.samples: List[KilterSample] = []
        self._static_cache: Optional[torch.Tensor] = None

        self._index_samples()

    @property
    def num_grades(self) -> int:
        return self.grade_max - self.grade_min + 1

    def _index_samples(self) -> None:
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        json_files.sort()

        for fname in json_files:
            stem = fname[:-5]
            json_path = os.path.join(self.data_dir, fname)
            npy_path = os.path.join(self.data_dir, stem + '.npy')
            if not os.path.exists(npy_path):
                continue

            with open(json_path, 'r') as f:
                meta = json.load(f)
            grade_v = meta.get('grade_v')

            if grade_v is None and not self.include_unknown:
                continue
            if grade_v is None:
                continue
            if grade_v < self.grade_min or grade_v > self.grade_max:
                continue

            self.samples.append(KilterSample(npy_path=npy_path, grade_v=int(grade_v)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.data_dir} with grade_v in [{self.grade_min}, {self.grade_max}].")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        arr = np.load(sample.npy_path)  # H x W x 6
        arr = np.transpose(arr, (2, 0, 1))  # C x H x W

        route = torch.from_numpy(arr[0:4]).float()
        static = torch.from_numpy(arr[4:6]).float()
        grade = torch.tensor(sample.grade_v - self.grade_min, dtype=torch.int64)
        return route, static, grade

    def get_static(self) -> torch.Tensor:
        """Return the static (hold_presence, hold_size) map as (2, H, W)."""
        if self._static_cache is None:
            route, static, _ = self[0]
            self._static_cache = static
        return self._static_cache

    def npy_paths(self, indices: Optional[Sequence[int]] = None) -> List[str]:
        if indices is None:
            return [s.npy_path for s in self.samples]
        return [self.samples[i].npy_path for i in indices]


def compute_pos_weight(
    npy_paths: Sequence[str],
    hold_channel: int = 4,
    route_channels: Tuple[int, int, int, int] = (0, 1, 2, 3),
) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss per route channel.
    pos_weight = (#neg / #pos) over hold positions only.
    """
    pos_counts = np.zeros(len(route_channels), dtype=np.float64)
    total_hold = 0.0

    for p in npy_paths:
        arr = np.load(p)
        hold = arr[..., hold_channel] > 0
        total_hold += float(hold.sum())
        for i, ch in enumerate(route_channels):
            pos_counts[i] += float((arr[..., ch] > 0).sum())

    neg_counts = np.maximum(total_hold - pos_counts, 0.0)
    pos_counts = np.maximum(pos_counts, 1.0)  # avoid divide-by-zero
    pos_weight = neg_counts / pos_counts
    return torch.tensor(pos_weight, dtype=torch.float32)
