"""Deterministic fixed monitoring derived from annotation eligibility only."""
from collections import Counter
import json

import numpy as np

from simlingo_training.lidar.data import TASKS
from simlingo_training.lidar.evaluation import (group_metrics, structured_answer,
                                              dreamer_decision)
from simlingo_training.lidar.validation import route_balanced
from simlingo_training.lidar.validation_run import identity


def build_protocol(protocol, baseline_rows, train):
    baseline = {identity(r): r for r in baseline_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError('Duplicate baseline identity')
    monitor = {task: list(items) for task, items in protocol['monitor'].items()}
    coverage = {}
    for task, parse in [('qa', structured_answer), ('dreamer', dreamer_decision)]:
        selected = {identity(r) for r in monitor[task]}
        def known(item):
            # Deliberately do not inspect predictions or their metric values.
            return parse(baseline[identity(item)]['reference']) is not None
        eligible = [item for item in monitor[task] if known(item)]
        groups = Counter(baseline[identity(r)]['group'] for r in eligible)
        candidates = [r for r in protocol['requests'][task] if identity(r) not in selected and known(r)]
        while len(eligible) < 100:
            if not candidates:
                raise ValueError('Fewer than 100 eligible targets: ' + task)
            item = min(candidates, key=lambda r: (groups[baseline[identity(r)]['group']],
                         baseline[identity(r)]['group'], r['uid'], r['seed']))
            candidates.remove(item)
            monitor[task].append(item); eligible.append(item)
            groups[baseline[identity(item)]['group']] += 1
        coverage[task] = {'scored_targets': len(eligible), 'groups': len(groups),
                          'requests': len(monitor[task])}
    keys = {identity(r) for items in monitor.values() for r in items}
    rgb_rows = [r for r in baseline_rows if identity(r) in keys]
    expected_rgb = {identity(r) for t, items in monitor.items() if t != 'geometry_qa' for r in items}
    if {identity(r) for r in rgb_rows} != expected_rgb:
        raise ValueError('Monitoring baseline is incomplete')
    diagnostics = {}
    for task in TASKS:
        selected = route_balanced(train.rows, train.pools[task], 32)
        if len(selected) != 32:
            raise ValueError('Training diagnostic pool too small: ' + task)
        diagnostics[task] = [{'index': i, 'task': task, 'replay': False,
             'seed': int(np.random.SeedSequence([42, j, 171]).generate_state(1)[0]),
             'position': j, 'uid': train.rows[i]['id']} for j, i in enumerate(selected)]
    train_ids = {r['id'] for r in train.rows}
    if any(r['uid'] in train_ids for items in monitor.values() for r in items):
        raise ValueError('Training/validation monitoring overlap')
    return {'schema': 1, 'seed': 42, 'manifest_sha256': train.manifest_hash,
            'requests': monitor, 'train_requests': diagnostics,
            'coverage': coverage, 'baseline_metrics': group_metrics(rgb_rows),
            'paired_requests': sum(map(len, monitor.values())), 'rgb_requests': len(rgb_rows),
            'recorded_rgb_seconds': sum(r['total_seconds'] for r in rgb_rows),
            'selection': 'Fixed annotation eligibility and group coverage; no prediction correctness used',
            'limits': 'Interim monitor only. Full held-out validation is required for retention.'}


def baseline_records(path):
    with path.open() as handle:
        binding = json.loads(next(handle))['binding']
        rows = [json.loads(line) for line in handle]
    return binding, rows
