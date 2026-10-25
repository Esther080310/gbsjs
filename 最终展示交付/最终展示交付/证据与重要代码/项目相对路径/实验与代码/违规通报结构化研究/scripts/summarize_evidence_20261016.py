"""Frozen paired company bootstrap for completed evidence-v1 experiments.

All years of a company share a resampling count, reused for every model/seed.
Weighted stable top-k exactly equals expanding sampled rows and reranking.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
E = P.parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def top10_recall(y, score, counts=None):
    order = np.argsort(-score, kind='stable')
    if counts is None:
        counts = np.ones(len(y), dtype=np.int64)
    weights = counts[order]
    positives = int(np.dot(counts, y))
    if not positives:
        return np.nan
    budget = int(math.ceil(.1 * int(weights.sum())))
    before = np.cumsum(weights) - weights
    selected_weights = np.minimum(weights, np.maximum(budget - before, 0))
    return float(np.dot(selected_weights, y[order]) / positives)


def bootstrap_pair(root, candidate, reference, years, seeds, repetitions=1000, random_seed=20261016):
    keys, labels, models = {}, {}, {}
    for year in years:
        with np.load(E / f'财务年度配对修正/cache/{year}.npz', allow_pickle=False) as z:
            keys[year] = z['3_symbol'].astype(str)
            labels[year] = z['3_y'].astype(np.int64)
        for seed in seeds:
            for arm in (candidate, reference):
                with np.load(root / str(year) / f'{arm}_s{seed}/predictions.npz', allow_pickle=False) as z:
                    models[year, seed, arm] = z['a_logit'].copy()
    companies = sorted(set().union(*(set(v) for v in keys.values())))
    company_index = {symbol: i for i, symbol in enumerate(companies)}
    row_company = {year: np.array([company_index[s] for s in symbols]) for year, symbols in keys.items()}
    orders = {key: np.argsort(-score, kind='stable') for key, score in models.items()}
    original_cells = []
    for year in years:
        for seed in seeds:
            difference = top10_recall(labels[year], models[year, seed, candidate]) - top10_recall(labels[year], models[year, seed, reference])
            original_cells.append({'year': year, 'seed': seed, 'recall_difference': difference,
                                   'recall_difference_pp': 100 * difference})
    yearly = {str(year): float(np.mean([c['recall_difference'] for c in original_cells if c['year'] == year])) for year in years}
    macro = float(np.mean([c['recall_difference'] for c in original_cells]))
    rng = np.random.default_rng(random_seed)
    probabilities = np.full(len(companies), 1 / len(companies))
    draws = np.empty(repetitions)
    for repeat in range(repetitions):
        counts = rng.multinomial(len(companies), probabilities)
        differences = []
        for year in years:
            weights = counts[row_company[year]]
            positive_count = int(np.dot(weights, labels[year]))
            budget = math.ceil(.1 * int(weights.sum()))
            for seed in seeds:
                recalls = []
                for arm in (candidate, reference):
                    order = orders[year, seed, arm]
                    w = weights[order]
                    before = np.cumsum(w) - w
                    selected = np.minimum(w, np.maximum(budget - before, 0))
                    recalls.append(float(np.dot(selected, labels[year][order]) / positive_count))
                differences.append(recalls[0] - recalls[1])
        draws[repeat] = np.mean(differences)
    lower, upper = np.quantile(draws, [.025, .975])
    per_seed = {str(seed): float(np.mean([c['recall_difference'] for c in original_cells if c['seed'] == seed])) for seed in seeds}
    return {'candidate': candidate, 'reference': reference, 'years': years, 'seeds': seeds,
            'cells': original_cells, 'year_mean_differences': yearly,
            'macro_mean_difference': macro, 'macro_mean_difference_pp': 100 * macro,
            'company_cluster_bootstrap': {'repetitions': repetitions, 'seed': random_seed,
                'companies': len(companies), 'percentile_95_lower': float(lower), 'percentile_95_upper': float(upper),
                'percentile_95_lower_pp': 100 * float(lower), 'percentile_95_upper_pp': 100 * float(upper),
                'cross_year_company_counts_shared': True, 'models_and_seeds_use_same_resample': True,
                'top10_reranked_each_resample': True, 'seed_then_year_average': True},
            'per_seed_two_year_macro_differences': per_seed,
            'seed_difference_min': min(per_seed.values()), 'seed_difference_max': max(per_seed.values()),
            'positive_seed_count': sum(v > 0 for v in per_seed.values()),
            'positive_year_seed_cells': sum(c['recall_difference'] > 0 for c in original_cells),
            'continue_rule': {'six_seeds_required': len(seeds) == 6, 'macro_at_least_half_pp': macro >= .005,
                'each_year_nonnegative': all(d >= 0 for d in yearly.values()), 'cluster_95_lower_positive': bool(lower > 0),
                'passed': len(seeds) == 6 and macro >= .005 and all(d >= 0 for d in yearly.values()) and lower > 0}}


def check_weighted_topk():
    y = np.array([1, 0, 1, 0, 1])
    score = np.array([.5, .5, .3, .1, .9])
    for counts in (np.array([3, 2, 0, 4, 1]), np.array([0, 0, 4, 3, 5])):
        ids = np.repeat(np.arange(len(y)), counts)
        assert top10_recall(y, score, counts) == top10_recall(y[ids], score[ids])


def summarize(root, pairs, repetitions=1000):
    assert (root / 'pilot_gate.json').exists() and (root / 'verification.json').exists()
    before_gate = hashlib.sha256((root / 'pilot_gate.json').read_bytes()).hexdigest()
    records = json.loads((root / 'runs.json').read_text())
    years = sorted({r['year'] for r in records})
    seeds = sorted({r['seed'] for r in records})
    metrics = pd.read_csv(root / 'test_metrics.csv')
    for row in metrics.to_dict('records'):
        year, seed, arm = row['year'], row['seed'], row['arm']
        with np.load(E / f'财务年度配对修正/cache/{year}.npz', allow_pickle=False) as z:
            y = z['3_y']
        with np.load(root / str(year) / f'{arm}_s{seed}/predictions.npz', allow_pickle=False) as z:
            recall = top10_recall(y, z['a_logit'])
        # Original evaluation divides float32 label sums; bootstrap uses exact integer counts.
        assert abs(recall - row['recall']) < 5e-8
    comparisons = [bootstrap_pair(root, a, b, years, seeds, repetitions) for a, b in pairs]
    columns = ['recall', 'precision', 'overall_ap', 'overall_logloss', 'overall_brier',
               'conditional_logloss', 'conditional_brier', 'type_hit2', 'type_macro_coverage']
    yearly = metrics.groupby(['year', 'arm'], sort=True)[columns].mean().reset_index()
    macro = metrics.groupby(['arm'], sort=True)[columns].mean().reset_index()
    yearly.to_csv(root / 'year_arm_means.csv', index=False)
    macro.to_csv(root / 'macro_arm_means.csv', index=False)
    result = {'runs': len(records), 'years': years, 'seeds': seeds,
              'macro_arm_means': macro.to_dict('records'), 'year_arm_means': yearly.to_dict('records'),
              'comparisons': comparisons, 'train_seconds_sum': sum(r['train_seconds'] for r in records),
              'runtime_context': 'wall-clock runtime may overlap separately authorized experiments; no speed comparison',
              'pilot_gate_sha256': before_gate, 'gate_not_changed_by_test_results': True,
              'interpretation': 'fixed models on reused development years; not independent-year validation'}
    assert hashlib.sha256((root / 'pilot_gate.json').read_bytes()).hexdigest() == before_gate
    write_json(root / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('route', choices=['semantic', 'financial'])
    parser.add_argument('--repetitions', type=int, default=1000)
    args = parser.parse_args()
    check_weighted_topk()
    root = P / 'artifacts/experiments/20261016-evidence-v1' / args.route
    pairs = [('C', 'A'), ('C', 'B')] if args.route == 'semantic' else [('FV', 'FM'), ('FV', 'F0')]
    result = summarize(root, pairs, args.repetitions)
    print(json.dumps({'route': args.route, 'runs': result['runs'],
        'comparisons': [{'pair': c['candidate'] + '-' + c['reference'], 'difference_pp': c['macro_mean_difference_pp'],
            'CI_pp': [c['company_cluster_bootstrap']['percentile_95_lower_pp'], c['company_cluster_bootstrap']['percentile_95_upper_pp']],
            'continue': c['continue_rule']['passed']} for c in result['comparisons']]}), flush=True)


if __name__ == '__main__':
    main()
