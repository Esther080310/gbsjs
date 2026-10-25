"""Build a frozen-cutoff 2022 evaluation cache and audit features for external replication.

Offline only: no training, no API. Writes exclusively to
artifacts/experiments/20261016-audit-year2022-v1/ and does not touch the frozen
2020/2021 caches, the frozen aligned tables, or the completed evidence-v1 experiment.

The audit feature definition, Chinese-label grouping, U0/UM/UA construction and
alignment logic are reused verbatim from the frozen implementation
(notice_pipeline/audit_information.py and notice_pipeline/csmar_alignment.py).
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P))
sys.path.insert(0, str(E / '财务年度配对修正'))

import numpy as np
import pandas as pd
import pipeline as corrected  # noqa: E402  财务年度配对修正/pipeline.py
from notice_pipeline import audit_information as AI  # noqa: E402
from notice_pipeline import csmar_alignment as AL  # noqa: E402

YEAR = 2022
OUT = P / 'artifacts/experiments/20261016-audit-year2022-v1'
ALIGNED = P / 'artifacts/followup/csmar/processed/20261016-v1/aligned'
NORMALIZED = P / 'artifacts/followup/csmar/processed/20261016-v1/normalized'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def export_cache(pack, dest):
    data = {}
    for fold, split in enumerate(pack['splits']):
        for key in ('industry', 'q', 'state', 'y', 'types'):
            data[f'{fold}_{key}'] = split[key]
        for j, values in enumerate(split['xs']):
            data[f'{fold}_x{j}'] = values
        data[f'{fold}_symbol'] = split['keys'].Symbol.to_numpy().astype(str)
        data[f'{fold}_year'] = split['keys'].economic_year.to_numpy()
    np.savez_compressed(dest, **data)
    with np.load(dest, allow_pickle=False) as reread:
        for key, expected in data.items():
            np.testing.assert_array_equal(reread[key], expected)
    return data


def main():
    (OUT / 'cache').mkdir(parents=True, exist_ok=True)
    (OUT / 'aligned').mkdir(parents=True, exist_ok=True)
    (OUT / 'features').mkdir(parents=True, exist_ok=True)
    protected = [ALIGNED / 'FIN_Audit_annual_candidates.csv.gz',
                 NORMALIZED / 'IAR_Rept.csv.gz',
                 E / '财务年度配对修正/data/x2_annual.csv.gz',
                 E / '财务年度配对修正/cache/2020.npz', E / '财务年度配对修正/cache/2021.npz']
    before = {str(p): sha256(p) for p in protected}

    # ---- 1. corrected sample and preprocessing, identical code path as 2020/2021 ----
    pack = corrected.prepare(YEAR, corrected=True)
    cache_path = OUT / 'cache' / f'{YEAR}.npz'
    export_cache(pack, cache_path)
    (OUT / 'cache' / f'{YEAR}.json').write_text(json.dumps({
        'names': pack['names'], 'meta': pack['meta'], 'robust_scale': pack['robust_scale'],
        'events': [list(pack['events'][0]), list(pack['events'][1])],
        'source': '财务年度配对修正/pipeline.py prepare(2022, corrected=True)',
        'fold_definitions': {'fold0': 'economic_year<=2019', 'fold1': '2020', 'fold2': '2021', 'fold3': '2022'},
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # ---- 2. annual alignment, FIN_Audit only (audit features need nothing else) ----
    csv.field_size_limit(128 * 1024 * 1024)
    pubs = AL.publication_table(AL.load(NORMALIZED / 'IAR_Rept.csv.gz'))
    candidates = pd.read_csv(ALIGNED / 'FIN_Audit_annual_candidates.csv.gz', dtype=str,
                             keep_default_na=False)
    for column in ('economic_year',):
        candidates[column] = candidates[column].astype('int64')
    for column in ('duplicate_count', 'candidate_count'):
        candidates[column] = candidates[column].astype('int64')
    samples = AL.cache_samples(cache_path, YEAR)
    aligned = AL.align_samples(samples, pubs, {'FIN_Audit': candidates})
    align_path = OUT / 'aligned' / f'sample_alignment_{YEAR}.csv.gz'
    aligned.to_csv(align_path, index=False, compression='gzip')

    # ---- 3. audit features, identical construction to the frozen 2020/2021 files ----
    assert AI.boundary_checks()
    candidate_rows = [{key: row[key] for key in ('Symbol', 'economic_year', 'Audittyp', 'Annodt', 'content_conflict')}
                      for row in candidates.to_dict('records')]
    groups = {fold: [] for fold in range(4)}
    for row in AI._rows(align_path):
        groups[int(row['fold'])].append(row)
    arrays = {'feature_names': np.asarray(AI.FEATURE_NAMES, dtype=str)}
    report = {'year': YEAR, 'feature_names': list(AI.FEATURE_NAMES), 'mapping': AI.LABEL_TO_GROUP,
              'groups': list(AI.GROUPS), 'append_node': 2, 'append_node_name': 'X3', 'dimensions': 7,
              'uses_text': False, 'uses_dates_as_input': False, 'retrospective_only': True,
              'r_definition': 'current-snapshot unique same-company-year aligned match; dates never modify r',
              'folds': []}
    unique_matched, unknown_rows = {}, []
    with np.load(cache_path, allow_pickle=False) as frozen:
        for fold, rows in groups.items():
            n = len(rows)
            np.testing.assert_array_equal(np.array([int(r['row_index']) for r in rows]), np.arange(n))
            symbols = np.array([r['Symbol'] for r in rows], dtype=str)
            years = np.array([int(r['economic_year']) for r in rows], dtype=np.int64)
            np.testing.assert_array_equal(symbols, frozen[f'{fold}_symbol'].astype(str))
            np.testing.assert_array_equal(years, frozen[f'{fold}_year'])
            np.testing.assert_array_equal(np.array([float(r['Y']) for r in rows], dtype=np.float32), frozen[f'{fold}_y'])
            np.testing.assert_array_equal(
                np.array([[float(r[f'type_{j}']) for j in range(10)] for r in rows], dtype=np.float32),
                frozen[f'{fold}_types'])
            labels, matched = [], np.zeros(n, dtype=bool)
            indices = np.full(n, -1, dtype=np.int64)
            iar_cohort, date_cohort = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
            for i, row in enumerate(rows):
                raw_index = row['FIN_Audit__source_index']
                index = int(float(raw_index)) if raw_index else -1
                if index < 0:
                    labels.append('')
                    continue
                if int(row['FIN_Audit__candidate_count']) != 1:
                    raise ValueError('Matched source is not unique')
                candidate = candidate_rows[index]
                assert (candidate['Symbol'], int(candidate['economic_year'])) == (symbols[i], years[i])
                assert candidate['content_conflict'].lower() != 'true'
                matched[i], indices[i] = True, index
                labels.append(candidate['Audittyp'])
                unique_matched[(symbols[i], int(years[i]))] = candidate['Audittyp']
                iar_cohort[i] = row['publication_gate'].lower() == 'true'
                audit_date, cutoff = AI._date(candidate['Annodt']), AI._date(row['scoring_cutoff'])
                date_cohort[i] = bool(iar_cohort[i] and audit_date and cutoff and audit_date <= cutoff)
            data, unknown = AI.variants(matched, labels)
            for name, values in data.items():
                arrays[f'{fold}_{name}'] = values
            arrays.update({f'{fold}_symbol': symbols, f'{fold}_year': years, f'{fold}_row_index': np.arange(n),
                           f'{fold}_source_index': indices, f'{fold}_audit_type': np.asarray(labels, dtype=str),
                           f'{fold}_unknown': unknown, f'{fold}_iar_cohort': iar_cohort,
                           f'{fold}_iar_and_auditdate_cohort': date_cohort,
                           f'{fold}_y': frozen[f'{fold}_y'].copy(), f'{fold}_types': frozen[f'{fold}_types'].copy()})
            count = Counter(labels[i] for i in np.flatnonzero(matched))
            report['folds'].append({'year': YEAR, 'fold': fold, 'rows': n, 'matched': int(matched.sum()),
                                    'unmatched': int((~matched).sum()), 'unknown_matched': int(unknown.sum()),
                                    'raw_label_counts': dict(count),
                                    'category_counts': dict(zip(AI.GROUPS, data['UA'][:, 1:].sum(0).astype(int).tolist())),
                                    'positives': int(frozen[f'{fold}_y'].sum()),
                                    'iar_common_cohort': int(iar_cohort.sum()),
                                    'iar_auditdate_common_cohort': int(date_cohort.sum())})
            for i in np.flatnonzero(unknown):
                unknown_rows.append({'year': YEAR, 'fold': fold, 'row_index': int(i),
                                     'symbol': symbols[i], 'economic_year': int(years[i]),
                                     'raw_audit_type': labels[i], 'source_index': int(indices[i])})
    feature_path = OUT / 'features' / f'audit_features_{YEAR}.npz'
    np.savez_compressed(feature_path, **arrays)
    with np.load(feature_path, allow_pickle=False) as reread:
        for key, expected in arrays.items():
            np.testing.assert_array_equal(reread[key], expected)
    # invariants
    for fold in range(4):
        ua, um, u0 = arrays[f'{fold}_UA'], arrays[f'{fold}_UM'], arrays[f'{fold}_U0']
        matched = arrays[f'{fold}_source_index'] >= 0
        assert np.array_equal(ua[:, 0], matched.astype(np.float32))
        assert np.array_equal(um[:, 0], matched.astype(np.float32)) and not um[:, 1:].any()
        assert not u0.any()
        assert np.array_equal(ua[:, 1:].sum(1), matched.astype(np.float32))
    report.update(unique_matched_company_years=len(unique_matched),
                  unique_raw_category_counts=dict(Counter(unique_matched.values())),
                  unknown_rows=unknown_rows, arrays_reloaded_exact=True,
                  cache_sha256=sha256(cache_path), alignment_sha256=sha256(align_path),
                  feature_sha256=sha256(feature_path),
                  source_sha256={str(p): before[str(p)] for p in protected})
    (OUT / 'features' / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                                    encoding='utf-8')
    after = {str(p): sha256(p) for p in protected}
    assert before == after, 'Protected frozen input changed'
    print(json.dumps({k: report[k] for k in ('year', 'unique_matched_company_years', 'cache_sha256',
                                             'feature_sha256', 'source_sha256')}, ensure_ascii=False, indent=2))
    for fold in report['folds']:
        print(f"fold{fold['fold']}: rows={fold['rows']} positives={fold['positives']} "
              f"matched={fold['matched']} unmatched={fold['unmatched']} categories={fold['category_counts']}")


if __name__ == '__main__':
    main()
