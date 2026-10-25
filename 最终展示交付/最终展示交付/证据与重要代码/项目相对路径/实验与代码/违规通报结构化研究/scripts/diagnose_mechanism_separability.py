"""Mechanism separability diagnostic: are LLM-extracted mechanisms distinguishable in X?

Offline only: no training, no API. Reads the existing curated fact-to-sample links,
attaches the LLM mechanism labels from the stored extraction responses, joins to the
frozen corrected cache, and reports coverage before any statistical test.

The question this must answer: among positive company-years, do different violation
mechanisms occupy distinguishable regions of (a) the model-free transformed inputs and
(b) the frozen S32 message representation? If neither, no mechanism-aware model can help.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))

import numpy as np
import pandas as pd

RUNS = P / 'artifacts/runs'
LINKS = P / 'artifacts/followup/event_sample_links.csv'
OUT = P / 'artifacts/experiments/20261016-mechanism-separability-v1'
CACHE = E / '财务年度配对修正/cache'


def load_mechanisms():
    """(record_id, fact_id) -> mechanism, from the stored extraction responses."""
    table = {}
    for run in RUNS.iterdir():
        if not run.is_dir():
            continue
        for name in ('extractions.jsonl', 'partials.jsonl'):
            path = run / name
            if not path.exists():
                continue
            for line in path.open():
                record = json.loads(line)
                rid = str(record['record_id'])
                for fact in record.get('extraction', {}).get('facts', []):
                    mechanism = fact.get('mechanism')
                    if mechanism:
                        table[(rid, str(fact.get('fact_id')))] = mechanism
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mechanisms = load_mechanisms()
    links = pd.read_csv(LINKS, dtype={'symbol': str, 'record_id': str, 'fact_id': str})
    links['mechanism'] = [mechanisms.get((r, f)) for r, f in zip(links.record_id, links.fact_id)]
    print(f'mechanism lookup keys: {len(mechanisms)}')
    print(f'links: {len(links)}, with LLM mechanism: {int(links.mechanism.notna().sum())}')
    print(f'admissible links: {int(links.admissible.sum())}, '
          f'admissible with mechanism: {int((links.admissible & links.mechanism.notna()).sum())}')
    print('\nmechanism x admissible:')
    print(links[links.admissible].mechanism.value_counts().to_string())

    usable = links[links.admissible & links.mechanism.notna()].copy()
    usable['symbol'] = usable.symbol.str.zfill(6)
    unit = usable.groupby(['evaluation_year', 'fold', 'symbol', 'economic_year', 'original_sample_y'])
    rows = []
    for key, group in unit:
        evaluation_year, fold, symbol, economic_year, y = key
        counts = Counter(group.mechanism)
        rows.append({'evaluation_year': int(evaluation_year), 'fold': int(fold), 'symbol': symbol,
                     'economic_year': int(economic_year), 'y': float(y),
                     'n_facts': int(len(group)), 'n_mechanisms': len(counts),
                     'dominant': counts.most_common(1)[0][0],
                     'dominant_share': counts.most_common(1)[0][1] / len(group),
                     'mechanisms': ';'.join(sorted(counts))})
    data = pd.DataFrame(rows)
    data.to_csv(OUT / 'mechanism_company_years.csv', index=False)
    print(f'\ncompany-year units: {len(data)}  (positives {int((data.y == 1).sum())})')
    print(f'duplicated across the two evaluation-year columns: '
          f'{len(data) - data.drop(columns=["evaluation_year", "fold"]).drop_duplicates().shape[0]}')
    unique = data.drop_duplicates(subset=['symbol', 'economic_year'])
    print(f'unique company-years (2020/2021 column deduplicated): {len(unique)}')
    print('\ndominant mechanism among unique positive company-years:')
    pos = unique[unique.y == 1]
    print(pos.dominant.value_counts().to_string())
    print(f'\npositives with a single mechanism: {int((pos.n_mechanisms == 1).sum())}')
    print(f'positives with dominant_share >= 0.6: {int((pos.dominant_share >= .6).sum())}')
    print('\nby economic_year:')
    print(pd.crosstab(pos.economic_year, pos.dominant).to_string())


if __name__ == '__main__':
    main()
