"""Fixed source-only FIN_Audit category information; no text or date input."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
import numpy as np

# Chinese export labels, checked against the local dictionary. Numeric era
# codes are intentionally never inferred. The five groups are research groups.
GROUPS = ('standard_unqualified', 'unqualified_with_explanation', 'qualified',
          'adverse', 'disclaimer_or_refusal', 'unknown')
LABEL_TO_GROUP = {
    '标准无保留意见': 0,
    '无保留意见加说明段': 1,
    '带有解释性说明的无保留意见': 1,
    '保留意见': 2,
    '保留意见加说明段': 2,
    '保留意见加事项段': 2,
    '否定意见': 3,
    '否定意见加说明段': 3,
    '无法表示意见': 4,
    '拒绝发表意见': 4,
}
FEATURE_NAMES = ('audit::unique_company_year_match',) + tuple('audit::' + group for group in GROUPS)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def variants(matched, labels):
    matched = np.asarray(matched, dtype=bool)
    if len(matched) != len(labels):
        raise ValueError('Match/label length differs')
    ua = np.zeros((len(matched), 7), dtype=np.float32)
    ua[:, 0] = matched
    unknown = np.zeros(len(matched), dtype=bool)
    for row in np.flatnonzero(matched):
        group = LABEL_TO_GROUP.get(str(labels[row]), 5)
        ua[row, 1 + group] = 1.
        unknown[row] = group == 5
    um = np.zeros_like(ua)
    um[:, 0] = matched
    return {'U0': np.zeros_like(ua), 'UM': um, 'UA': ua}, unknown


def _rows(path):
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def _date(value):
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def boundary_checks():
    labels = list(LABEL_TO_GROUP) + ['', 'not_a_known_label', '1', '标准无保留意见']
    matched = np.array([True] * (len(labels) - 1) + [False])
    data, unknown = variants(matched, labels)
    assert data['UA'].shape == (len(labels), 7) and data['UA'].dtype == np.float32
    assert np.array_equal(data['UM'][:, 0], matched)
    assert not data['UM'][:, 1:].any() and not data['U0'].any()
    assert np.array_equal(data['UA'][:, 1:].sum(1), matched.astype(np.float32))
    assert unknown.sum() == 3 and not unknown[-1]
    assert data['UA'][0, 1] == 1 and not data['UA'][-1].any()
    return True


def prepare(project_root):
    project_root = Path(project_root).resolve()
    aligned = project_root / 'artifacts/followup/csmar/processed/20261016-v1/aligned'
    out = project_root / 'artifacts/experiments/20261016-evidence-v1/audit_information/features'
    out.mkdir(parents=True, exist_ok=True)
    sources = [aligned / 'FIN_Audit_annual_candidates.csv.gz', project_root / 'artifacts/followup/csmar/字段字典汇总.csv',
               project_root / 'artifacts/experiments/20261016-evidence-v1/audit_source_quality/summary.json']
    sources += [aligned / f'sample_alignment_{year}.csv.gz' for year in (2020, 2021)]
    sources += [project_root.parent / f'财务年度配对修正/cache/{year}.npz' for year in (2020, 2021)]
    before = {str(path): sha256(path) for path in sources}
    assert boundary_checks()
    csv.field_size_limit(128 * 1024 * 1024)
    candidates = []
    for row in _rows(sources[0]):
        # Only audited category/key metadata are retained. No audit body is used.
        candidates.append({key: row[key] for key in ('Symbol', 'economic_year', 'Audittyp', 'Annodt', 'content_conflict')})
    report = {'feature_names': FEATURE_NAMES, 'mapping': LABEL_TO_GROUP, 'groups': GROUPS,
              'source_sha256': before, 'append_node': 2, 'append_node_name': 'X3', 'dimensions': 7,
              'uses_text': False, 'uses_dates_as_input': False, 'uses_correction_mask': False,
              'r_definition': 'current-snapshot unique same-company-year aligned match; dates never modify r',
              'historical_version_verified': False, 'retrospective_only': True, 'boundary_checks_passed': True, 'folds': []}
    unique_matched, unknown_rows = {}, []
    for year in (2020, 2021):
        groups = {f: [] for f in range(4)}
        for row in _rows(aligned / f'sample_alignment_{year}.csv.gz'):
            groups[int(row['fold'])].append(row)
        arrays = {'feature_names': np.asarray(FEATURE_NAMES, dtype=str)}
        with np.load(project_root.parent / f'财务年度配对修正/cache/{year}.npz', allow_pickle=False) as frozen:
            for fold, rows in groups.items():
                n = len(rows)
                np.testing.assert_array_equal(np.array([int(r['row_index']) for r in rows]), np.arange(n))
                symbols = np.array([r['Symbol'] for r in rows], dtype=str)
                years = np.array([int(r['economic_year']) for r in rows], dtype=np.int64)
                np.testing.assert_array_equal(symbols, frozen[f'{fold}_symbol'].astype(str))
                np.testing.assert_array_equal(years, frozen[f'{fold}_year'])
                np.testing.assert_array_equal(np.array([float(r['Y']) for r in rows], dtype=np.float32), frozen[f'{fold}_y'])
                np.testing.assert_array_equal(np.array([[float(r[f'type_{j}']) for j in range(10)] for r in rows], dtype=np.float32), frozen[f'{fold}_types'])
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
                    candidate = candidates[index]
                    assert (candidate['Symbol'], int(candidate['economic_year'])) == (symbols[i], years[i])
                    assert candidate['content_conflict'].lower() != 'true'
                    matched[i], indices[i] = True, index
                    labels.append(candidate['Audittyp'])
                    unique_matched[(symbols[i], int(years[i]))] = candidate['Audittyp']
                    # Date metadata provide common sensitivity cohorts only.
                    iar_cohort[i] = row['publication_gate'].lower() == 'true'
                    audit_date, cutoff = _date(candidate['Annodt']), _date(row['scoring_cutoff'])
                    date_cohort[i] = bool(iar_cohort[i] and audit_date and cutoff and audit_date <= cutoff)
                data, unknown = variants(matched, labels)
                for name, values in data.items():
                    arrays[f'{fold}_{name}'] = values
                arrays.update({f'{fold}_symbol': symbols, f'{fold}_year': years,
                    f'{fold}_row_index': np.arange(n), f'{fold}_source_index': indices,
                    f'{fold}_audit_type': np.asarray(labels, dtype=str), f'{fold}_unknown': unknown,
                    f'{fold}_iar_cohort': iar_cohort, f'{fold}_iar_and_auditdate_cohort': date_cohort,
                    f'{fold}_y': frozen[f'{fold}_y'].copy(), f'{fold}_types': frozen[f'{fold}_types'].copy()})
                count = Counter(labels[i] for i in np.flatnonzero(matched))
                report['folds'].append({'year': year, 'fold': fold, 'rows': n, 'matched': int(matched.sum()),
                    'unmatched': int((~matched).sum()), 'unknown_matched': int(unknown.sum()),
                    'raw_label_counts': dict(count), 'category_counts': dict(zip(GROUPS, data['UA'][:, 1:].sum(0).astype(int).tolist())),
                    'iar_common_cohort': int(iar_cohort.sum()), 'iar_auditdate_common_cohort': int(date_cohort.sum())})
                for i in np.flatnonzero(unknown):
                    unknown_rows.append({'year': year, 'fold': fold, 'row_index': int(i), 'symbol': symbols[i],
                                         'economic_year': int(years[i]), 'raw_audit_type': labels[i], 'source_index': int(indices[i])})
        dest = out / f'audit_features_{year}.npz'
        np.savez_compressed(dest, **arrays)
        with np.load(dest, allow_pickle=False) as reread:
            for key, expected in arrays.items():
                np.testing.assert_array_equal(reread[key], expected)
    report.update(unique_matched_company_years=len(unique_matched), unique_raw_category_counts=dict(Counter(unique_matched.values())),
                  unknown_rows=unknown_rows, arrays_reloaded_exact=True,
                  feature_sha256={str(out / f'audit_features_{y}.npz'): sha256(out / f'audit_features_{y}.npz') for y in (2020, 2021)})
    assert before == {str(path): sha256(path) for path in sources}
    (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def load_audit_features(output_dir, year, fold, variant):
    if variant not in ('U0', 'UM', 'UA'):
        raise ValueError('Unknown audit variant')
    with np.load(Path(output_dir) / f'audit_features_{year}.npz', allow_pickle=False) as pack:
        return pack[f'{fold}_{variant}'].copy()
