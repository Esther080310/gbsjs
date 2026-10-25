"""Post-gate paired company bootstrap for the 2022 external replication.

Reuses the frozen metric and weighted-top-k helpers. The paired company bootstrap is
the same construction as summarize_audit_distributed.compare: companies are resampled
with replacement, every model and seed shares the draw, and the 10% list is re-ranked
inside each draw before averaging seeds (and years, when more than one).

The capacity is not chosen here: 128 was frozen by the 2020/2021 fold1 gate, and this
script must not select anything on 2022 fold3.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_evidence_20261016 import top10_recall, check_weighted_topk

P = Path(__file__).resolve().parents[1]
OUT = P / 'artifacts/experiments/20261016-audit-year2022-v1'
ROOT = OUT / 'distributed'
CACHE = OUT / 'cache'
YEAR = 2022
SELECTED = 128


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n',
                    encoding='utf-8')


def available_seeds(records, budget, arm):
    return sorted({r['seed'] for r in records if r['budget'] == budget and r['arm'] == arm})


def compare(candidate, reference, seeds, repeats=1000, seed=20261016):
    with np.load(CACHE / f'{YEAR}.npz', allow_pickle=False) as frozen:
        symbols, labels = frozen['3_symbol'].astype(str), frozen['3_y'].astype(np.int64)
    scores, orders = {}, {}
    for model_seed in seeds:
        for budget, arm in (candidate, reference):
            path = ROOT / f'D{budget}' / f'{arm}_s{model_seed}' / 'predictions.npz'
            with np.load(path, allow_pickle=False) as saved:
                scores[model_seed, budget, arm] = saved['a_logit'].copy()
            orders[model_seed, budget, arm] = np.argsort(-scores[model_seed, budget, arm], kind='stable')
    companies = sorted(set(symbols))
    index = {symbol: i for i, symbol in enumerate(companies)}
    row_company = np.array([index[s] for s in symbols])
    cells = []
    for model_seed in seeds:
        c = top10_recall(labels, scores[model_seed, *candidate])
        r = top10_recall(labels, scores[model_seed, *reference])
        cells.append({'seed': model_seed, 'candidate_recall': c, 'reference_recall': r,
                      'difference': c - r, 'difference_pp': 100 * (c - r)})
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(companies), 1 / len(companies))
    draws = np.empty(repeats)
    for rep in range(repeats):
        company_counts = rng.multinomial(len(companies), probabilities)
        weights = company_counts[row_company]
        positives = int(np.dot(weights, labels))
        budget_count = math.ceil(.1 * int(weights.sum()))
        differences = []
        for model_seed in seeds:
            recalls = []
            for model in (candidate, reference):
                order = orders[model_seed, *model]
                w = weights[order]
                before = np.cumsum(w) - w
                selected = np.minimum(w, np.maximum(budget_count - before, 0))
                recalls.append(float(np.dot(selected, labels[order]) / positives))
            differences.append(recalls[0] - recalls[1])
        draws[rep] = float(np.mean(differences))
    lo, hi = np.quantile(draws, [.025, .975])
    per_seed = {str(s): float(np.mean([c['difference'] for c in cells if c['seed'] == s])) for s in seeds}
    macro = float(np.mean([c['difference'] for c in cells]))
    return {'candidate': {'budget': candidate[0], 'arm': candidate[1]},
            'reference': {'budget': reference[0], 'arm': reference[1]}, 'year': YEAR, 'seeds': list(seeds),
            'cells': cells, 'macro_mean_difference': macro, 'macro_mean_difference_pp': 100 * macro,
            'per_seed_macro_differences_pp': {k: 100 * v for k, v in per_seed.items()},
            'positive_seed_count': int(sum(v > 0 for v in per_seed.values())),
            'seed_difference_range_pp': [100 * min(per_seed.values()), 100 * max(per_seed.values())],
            'company_cluster_bootstrap': {'repetitions': repeats, 'seed': seed, 'companies': len(companies),
                'percentile_95_lower': float(lo), 'percentile_95_upper': float(hi),
                'percentile_95_lower_pp': 100 * float(lo), 'percentile_95_upper_pp': 100 * float(hi),
                'same_draw_all_models_and_seeds': True, 'top10_reranked_each_draw': True,
                'seed_mean_then_draw_mean': True, 'single_evaluation_year': True},
            'cluster_CI_lower_positive': bool(lo > 0), 'macro_positive': bool(macro > 0)}


def main():
    assert (ROOT / 'verification.json').exists(), 'training not complete'
    assert (ROOT / 'pilot_gate.json').exists(), 'gate missing'
    check_weighted_topk()
    records = json.loads((ROOT / 'runs.json').read_text())
    gate = json.loads((ROOT / 'pilot_gate.json').read_text())
    seeds = available_seeds(records, SELECTED, 'UA')
    for budget, arm in ((SELECTED, 'UA'), (SELECTED, 'UM'), (SELECTED, 'U0'), (32, 'U0')):
        assert available_seeds(records, budget, arm) == seeds, (budget, arm)
    has_32_ua = available_seeds(records, 32, 'UA') == seeds
    comparisons = {
        'total_utility_D128_UA_minus_D32_U0': compare((SELECTED, 'UA'), (32, 'U0'), seeds),
        'information_within_D128_UA_minus_UM': compare((SELECTED, 'UA'), (SELECTED, 'UM'), seeds),
        'capacity_D128_U0_minus_D32_U0': compare((SELECTED, 'U0'), (32, 'U0'), seeds),
    }
    if has_32_ua:
        comparisons['secondary_same_information_D128_UA_minus_D32_UA'] = compare((SELECTED, 'UA'), (32, 'UA'), seeds)
    total = comparisons['total_utility_D128_UA_minus_D32_U0']
    info = comparisons['information_within_D128_UA_minus_UM']
    replication = {'seeds': seeds, 'seed_count': len(seeds),
                   'rule': 'macro mean > 0 and cluster CI lower > 0 for total utility AND cluster CI lower > 0 for UA-UM',
                   'total_utility_interval_lower_positive': total['cluster_CI_lower_positive'],
                   'information_interval_lower_positive': info['cluster_CI_lower_positive'],
                   'replicated': bool(total['macro_positive'] and total['cluster_CI_lower_positive']
                                      and info['cluster_CI_lower_positive']),
                   'pilot_gate_passed': bool(gate['passed'])}
    metrics = pd.read_csv(ROOT / 'test_metrics.csv')
    fields = [f for f in ['recall', 'precision', 'overall_ap', 'overall_logloss', 'overall_brier',
                          'conditional_logloss', 'conditional_brier', 'type_hit2', 'type_macro_coverage']
              if f in metrics.columns]
    by_arm = metrics.groupby(['budget', 'arm'])[fields].mean().reset_index()
    by_arm.to_csv(ROOT / 'macro_budget_arm_means.csv', index=False)
    by_seed = metrics.groupby(['budget', 'arm', 'seed'])[fields].mean().reset_index()
    by_seed.to_csv(ROOT / 'per_seed_budget_arm_means.csv', index=False)
    result = {'year': YEAR, 'selected_capacity': SELECTED, 'seeds': seeds,
              'pilot_gate': gate, 'replication_decision': replication,
              'comparisons': comparisons,
              'message_bytes_by_budget': {32: 128, 128: 512}}
    write_json(ROOT / 'summary.json', result)
    print(json.dumps(replication, ensure_ascii=False, indent=2))
    for name, comp in comparisons.items():
        print(f"{name}: macro {comp['macro_mean_difference_pp']:+.3f} pp "
              f"CI [{comp['company_cluster_bootstrap']['percentile_95_lower_pp']:+.3f}, "
              f"{comp['company_cluster_bootstrap']['percentile_95_upper_pp']:+.3f}] "
              f"per-seed {[round(v,3) for v in comp['per_seed_macro_differences_pp'].values()]}")
    print(by_arm.to_string(index=False))


if __name__ == '__main__':
    main()
