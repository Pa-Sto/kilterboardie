"""Convert a verified Boardsesh snapshot to the existing ten-channel dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

CHANNELS = ['start', 'finish', 'hand', 'foot', 'hold_presence', 'hold_size',
            'orient_sin1', 'orient_cos1', 'orient_sin2', 'orient_cos2']
ROLES = {12: 0, 14: 1, 13: 2, 15: 3}


def decode_frames(frames, mapping, shape):
    if not frames or re.fullmatch(r'(?:p\d+r\d+)+', frames) is None:
        raise ValueError('invalid_or_multiframe_encoding')
    route = np.zeros((*shape, 4), dtype=np.float32)
    occupied = set()
    for placement, role in re.findall(r'p(\d+)r(\d+)', frames):
        placement, role = int(placement), int(role)
        if role not in ROLES:
            raise ValueError('unknown_role')
        if placement not in mapping:
            raise ValueError('unmapped_placement')
        row, col = mapping[placement]
        if (row, col) in occupied:
            raise ValueError('repeated_placement')
        occupied.add((row, col))
        route[row, col, ROLES[role]] = 1
    counts = route.sum(axis=(0, 1))
    if not 1 <= counts[0] <= 2 or not 1 <= counts[1] <= 2:
        raise ValueError('invalid_start_finish_counts')
    return route


def read_db(path):
    conn = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='runs/boardsesh_download/kilter-original-20260930.db')
    parser.add_argument('--hardware', default='runs/boardsesh_mapping_audit/hardware.db')
    parser.add_argument('--mapping', default='runs/boardsesh_mapping_audit/mapping_audit.json')
    parser.add_argument('--reference-dir', default='ImageData/50Degree/ExportClean')
    parser.add_argument('--output-dir', default='ImageData/50Degree/ExportBoardsesh')
    parser.add_argument('--angle', type=int, default=50)
    parser.add_argument('--grade-min', type=int, default=3)
    parser.add_argument('--grade-max', type=int, default=13)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output must be empty to prevent stale samples or overwrites.')
    out.mkdir(parents=True, exist_ok=True)
    audit_map = json.loads(Path(args.mapping).read_text())
    entries = audit_map['mapping']
    if len(entries) != 476 or not all(x['accepted'] for x in entries):
        raise RuntimeError('Expected the verified 476-hold mapping.')
    mapping = {x['placement_id']: (x['row'], x['col']) for x in entries}
    if len(mapping) != 476 or len(set(mapping.values())) != 476:
        raise RuntimeError('Mapping is not one-to-one.')
    references = {}
    reference_names = {}
    static = None
    for path in sorted(Path(args.reference_dir).glob('*.npy')):
        matrix = np.load(path)
        if matrix.shape != (34, 35, 10):
            raise RuntimeError('Unexpected reference tensor shape.')
        if static is None:
            static = matrix[..., 4:].copy()
        elif not np.array_equal(static, matrix[..., 4:]):
            raise RuntimeError('Reference static channels are inconsistent.')
        fingerprint = hashlib.sha256(matrix[..., :4].tobytes()).hexdigest()
        references.setdefault(fingerprint, []).append(path.stem)
        metadata = json.loads(path.with_suffix('.json').read_text())
        name = (metadata.get('name') or '').strip().casefold()
        reference_names.setdefault(name, []).append((path.stem, fingerprint))
    if static is None or set(map(tuple, np.argwhere(static[..., 0] > .5))) != set(mapping.values()):
        raise RuntimeError('Mapping does not match the static hold-presence mask.')
    hardware = read_db(args.hardware)
    grades = {}
    for row in hardware.execute("SELECT difficulty,boulder_name FROM board_difficulty_grades WHERE board_type='kilter'"):
        match = re.search(r'/V(\d+)$', row['boulder_name'])
        if match:
            grades[int(row['difficulty'])] = (int(match[1]), row['boulder_name'])
    hardware.close()
    db = read_db(args.database)
    query = '''SELECT c.*,s.display_difficulty,s.difficulty_average,
               s.ascensionist_count,s.quality_average FROM board_climbs c
               JOIN board_climb_stats s ON s.climb_uuid=c.uuid AND s.board_type=c.board_type
               WHERE c.board_type='kilter' AND c.layout_id=1 AND s.angle=?
               ORDER BY COALESCE(s.ascensionist_count,0) DESC,c.uuid'''
    counts = Counter()
    distribution = Counter()
    seen = {}
    duplicates = []
    named_checks = []
    exact_references = set()
    for row in db.execute(query, (args.angle,)):
        counts['angle_records'] += 1
        reason = None
        if row['is_draft'] or not row['is_listed'] or row['is_hidden']:
            reason = 'not_public_or_hidden'
        elif row['frames_count'] != 1:
            reason = 'multiframe'
        elif 10 not in json.loads(row['compatible_size_ids'] or '[]'):
            reason = 'incompatible_size'
        elif row['missing_hold_count']:
            reason = 'source_missing_holds'
        difficulty = row['display_difficulty']
        # Display difficulty is the catalog's discrete grade; never round an average.
        if reason is None and (difficulty is None or difficulty not in grades):
            reason = 'missing_or_unknown_display_grade'
        if reason:
            counts[reason] += 1
            continue
        grade, grade_raw = grades[difficulty]
        if not args.grade_min <= grade <= args.grade_max:
            counts['outside_grade_range'] += 1
            continue
        try:
            route = decode_frames(row['frames'], mapping, static.shape[:2])
        except ValueError as error:
            counts[str(error)] += 1
            continue
        fingerprint = hashlib.sha256(route.tobytes()).hexdigest()
        exact_references.update(references.get(fingerprint, []))
        for stem, expected in reference_names.get(row['name'].strip().casefold(), []):
            named_checks.append(dict(reference=stem, uuid=row['uuid'], name=row['name'],
                                    exact_match=expected == fingerprint))
        if fingerprint in seen:
            previous = seen[fingerprint]
            duplicates.append(dict(uuid=row['uuid'], kept_uuid=previous['uuid'],
                                   grade_v=grade, kept_grade_v=previous['grade_v']))
            counts['duplicate_layout'] += 1
            continue
        seen[fingerprint] = dict(uuid=row['uuid'], grade_v=grade)
        matrix = np.concatenate([route, static], axis=2)
        # A deterministic safe filename; original UUID remains in metadata.
        stem = 'boardsesh_' + hashlib.sha256(row['uuid'].encode()).hexdigest()[:24]
        metadata = dict(filename=stem+'.npy', rows=34, cols=35, channels=CHANNELS,
                        grade_v=grade, grade_raw=grade_raw, name=row['name'],
                        setter=row['setter_username'], angle=args.angle,
                        ring_counts=dict(zip(CHANNELS[:4], route.sum((0,1)).astype(int).tolist())),
                        source='boardsesh', source_uuid=row['uuid'], layout_id=1,
                        product_size_id=10, frames=row['frames'],
                        display_difficulty=difficulty, difficulty_average=row['difficulty_average'],
                        ascensionist_count=row['ascensionist_count'], quality_average=row['quality_average'],
                        hold_fingerprint=fingerprint, validation_errors=[])
        np.save(out/(stem+'.npy'), matrix)
        (out/(stem+'.json')).write_text(json.dumps(metadata, indent=2)+'\n')
        counts['exported'] += 1
        distribution[grade] += 1
    db.close()
    report = dict(config=vars(args), counts=dict(counts), grade_counts=dict(sorted(distribution.items())),
                  exact_reference_matches=len(exact_references), named_route_checks=named_checks,
                  duplicates=duplicates, duplicate_grade_conflicts=sum(x['grade_v']!=x['kept_grade_v'] for x in duplicates),
                  deduplication='Same role-aware matrix: retain highest ascensionist_count, then UUID order.',
                  static_channels='Copied exactly from validated ExportClean reference tensors.')
    (out/'dataset_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('named_route_checks','duplicates')},indent=2))


if __name__ == '__main__':
    main()
