"""Shared, leakage-resistant dataset split for the three-model benchmark."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from torch.utils.data import Subset


def make_split(data_dir, output, seed=42):
    groups = defaultdict(list)
    for path in sorted(Path(data_dir).glob('*.npy')):
        meta = json.loads(path.with_suffix('.json').read_text())
        a = np.load(path)[..., :4]
        # Keep role variants of the same selected holds in the same partition.
        key = hashlib.sha256((a.sum(2) > 0).tobytes()).hexdigest()
        groups[key].append((path.stem, int(meta['grade_v'])))
    buckets = defaultdict(list)
    for key, records in sorted(groups.items()):
        grade = Counter(g for _, g in records).most_common(1)[0][0]
        buckets[grade].append((key, records))
    rng = random.Random(seed)
    splits = {name: [] for name in ('train', 'val', 'test')}
    group_splits = {}
    grades = {name: Counter() for name in splits}
    for grade, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        n = max(1, round(len(bucket) * .1))
        for name, portion in zip(splits, (bucket[2*n:], bucket[:n], bucket[n:2*n])):
            for key, records in portion:
                group_splits[key] = name
                for stem, g in records:
                    splits[name].append(stem)
                    grades[name][g] += 1
    for names in splits.values():
        names.sort()
    result = dict(data_dir=str(Path(data_dir).resolve()), seed=seed, splits=splits,
                  grade_counts={k:dict(v) for k,v in grades.items()},
                  group_policy='identical hold occupancy, regardless of roles', groups=group_splits)
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    return result


def manifest_subsets(dataset, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    index = {Path(s.npy_path).stem: i for i, s in enumerate(dataset.samples)}
    sets = {k:set(v) for k,v in manifest['splits'].items()}
    if set(sets) != {'train', 'val', 'test'}:
        raise ValueError('Expected train, val, and test partitions.')
    if any(len(sets[k]) != len(v) for k, v in manifest['splits'].items()):
        raise ValueError('Duplicate sample in split partition.')
    if any(sets[a] & sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Split partitions overlap.')
    if set.union(*sets.values()) != set(index):
        raise ValueError('Split does not cover exactly this dataset.')
    return tuple(Subset(dataset, [index[x] for x in manifest['splits'][k]])
                 for k in ('train','val','test'))
