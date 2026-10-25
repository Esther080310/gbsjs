"""Post-gate paired company bootstrap for the selected audit capacity.

All capacity choices remain those frozen from fold1; this summary cannot choose
another budget using fold3. Other capacities keep their pilot status.
"""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import pandas as pd
from summarize_evidence_20261016 import top10_recall, check_weighted_topk

P = Path(__file__).resolve().parents[1]
E = P.parent
ROOT = P / 'artifacts/experiments/20261016-evidence-v1/audit_information/distributed'
FEATURES = ROOT.parent / 'features'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x)) + '\n', encoding='utf-8')


def compare(root, candidate, reference, years, seeds, repeats=1000, seed=20261016):
    labels, symbols, scores, orders = {}, {}, {}, {}
    for year in years:
        with np.load(E / f'财务年度配对修正/cache/{year}.npz', allow_pickle=False) as frozen:
            symbols[year], labels[year] = frozen['3_symbol'].astype(str), frozen['3_y'].astype(np.int64)
        for model_seed in seeds:
            for budget, arm in (candidate, reference):
                key = (year, model_seed, budget, arm)
                with np.load(root / f'D{budget}' / str(year) / f'{arm}_s{model_seed}/predictions.npz', allow_pickle=False) as saved:
                    scores[key] = saved['a_logit'].copy()
                orders[key] = np.argsort(-scores[key], kind='stable')
    companies = sorted(set().union(*(set(v) for v in symbols.values())))
    index = {symbol: i for i, symbol in enumerate(companies)}
    row_company = {year: np.array([index[s] for s in values]) for year, values in symbols.items()}
    cells = []
    for year in years:
        for model_seed in seeds:
            c = top10_recall(labels[year], scores[year, model_seed, *candidate])
            r = top10_recall(labels[year], scores[year, model_seed, *reference])
            cells.append({'year': year, 'seed': model_seed, 'candidate_recall': c, 'reference_recall': r,
                          'difference': c - r, 'difference_pp': 100 * (c - r)})
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(companies), 1 / len(companies))
    draws = np.empty(repeats)
    for rep in range(repeats):
        company_counts = rng.multinomial(len(companies), probabilities)
        differences = []
        for year in years:
            weights = company_counts[row_company[year]]
            positives = int(np.dot(weights, labels[year]))
            budget_count = math.ceil(.1 * int(weights.sum()))
            for model_seed in seeds:
                recalls = []
                for model in (candidate, reference):
                    order = orders[year, model_seed, *model]
                    w = weights[order]
                    before = np.cumsum(w) - w
                    selected = np.minimum(w, np.maximum(budget_count - before, 0))
                    recalls.append(float(np.dot(selected, labels[year][order]) / positives))
                differences.append(recalls[0] - recalls[1])
        draws[rep] = np.mean(differences)
    lo, hi = np.quantile(draws, [.025, .975])
    yearly = {str(year): float(np.mean([r['difference'] for r in cells if r['year'] == year])) for year in years}
    per_seed = {str(s): float(np.mean([r['difference'] for r in cells if r['seed'] == s])) for s in seeds}
    macro = float(np.mean([r['difference'] for r in cells]))
    return {'candidate': {'budget': candidate[0], 'arm': candidate[1]},
            'reference': {'budget': reference[0], 'arm': reference[1]}, 'years': years, 'seeds': seeds,
            'cells': cells, 'year_mean_differences': yearly, 'macro_mean_difference': macro,
            'macro_mean_difference_pp': 100 * macro, 'per_seed_macro_differences': per_seed,
            'positive_seed_count': sum(v > 0 for v in per_seed.values()),
            'seed_difference_range': [min(per_seed.values()), max(per_seed.values())],
            'company_cluster_bootstrap': {'repetitions': repeats, 'seed': seed, 'companies': len(companies),
                'percentile_95_lower': float(lo), 'percentile_95_upper': float(hi),
                'percentile_95_lower_pp': 100 * float(lo), 'percentile_95_upper_pp': 100 * float(hi),
                'same_company_cross_year_count': True, 'same_draw_all_models_and_seeds': True,
                'top10_reranked_each_draw': True, 'seed_then_year_mean': True},
            'stable_benefit_rule': {'macro_half_pp': macro >= .005, 'each_year_nonnegative': all(v >= 0 for v in yearly.values()),
                'cluster_CI_lower_positive': bool(lo > 0), 'six_seeds': len(seeds) == 6,
                'passed': len(seeds) == 6 and macro >= .005 and all(v >= 0 for v in yearly.values()) and lo > 0,
                'is_execution_prerequisite_for_further_distributed_work': False}}


def cohorts(root, records, budgets):
    rows = []
    for record in records:
        if record['budget'] not in budgets:
            continue
        year, model_seed, budget, arm = record['year'], record['seed'], record['budget'], record['arm']
        with np.load(FEATURES / f'audit_features_{year}.npz', allow_pickle=False) as feature:
            y = feature['3_y'].astype(np.int64)
            masks = {'matched_IAR': feature['3_iar_cohort'], 'matched_IAR_auditdate': feature['3_iar_and_auditdate_cohort']}
        with np.load(root / f'D{budget}' / str(year) / f'{arm}_s{model_seed}/predictions.npz', allow_pickle=False) as saved:
            score = saved['a_logit']
        full_order = np.argsort(-score, kind='stable')
        full_selected = np.zeros(len(y), dtype=bool)
        full_selected[full_order[:math.ceil(.1 * len(y))]] = True
        for name, mask in masks.items():
            found = int(y[full_selected & mask].sum())
            positives = int(y[mask].sum())
            rows.append({'year': year, 'seed': model_seed, 'budget': budget, 'arm': arm, 'cohort': name,
                         'rows': int(mask.sum()), 'positives': positives,
                         'reranked_within_cohort_recall': top10_recall(y[mask], score[mask]),
                         'fixed_full_list_cohort_found': found,
                         'fixed_full_list_cohort_recall': found / positives,
                         'fixed_full_list_cohort_selected': int((full_selected & mask).sum()),
                         'input_unchanged_by_cohort': True, 'historical_version_verified': False})
    pd.DataFrame(rows).to_csv(root / 'common_cohort_sensitivity.csv', index=False)


def main():
    assert (ROOT / 'verification.json').exists()
    check_weighted_topk()
    gate_before = hashlib.sha256((ROOT / 'pilot_gate.json').read_bytes()).hexdigest()
    selection_before = hashlib.sha256((ROOT / 'capacity_selection.json').read_bytes()).hexdigest()
    selection = json.loads((ROOT / 'capacity_selection.json').read_text())
    selected = selection['selected_budget']
    records = json.loads((ROOT / 'runs.json').read_text())
    years = sorted({r['year'] for r in records})
    seeds = sorted(set.intersection(*[
        {r['seed'] for r in records if r['budget'] == budget and r['arm'] == arm and r['year'] == year}
        for budget, arm in ((selected, 'UA'), (selected, 'UM'), (selected, 'U0'), (32, 'U0')) for year in years]))
    comparisons = {
        'total_utility_selected_UA_minus_D32_U0': compare(ROOT, (selected, 'UA'), (32, 'U0'), years, seeds),
        'information_within_selected_capacity_UA_minus_UM': compare(ROOT, (selected, 'UA'), (selected, 'UM'), years, seeds),
        'capacity_selected_U0_minus_D32_U0': compare(ROOT, (selected, 'U0'), (32, 'U0'), years, seeds),
    }
    metrics = pd.read_csv(ROOT / 'test_metrics.csv')
    fields = ['recall', 'precision', 'overall_ap', 'overall_logloss', 'overall_brier', 'conditional_logloss',
              'conditional_brier', 'type_hit2', 'type_macro_coverage']
    yearly = metrics.groupby(['budget', 'year', 'arm'])[fields].mean().reset_index()
    macro = metrics.groupby(['budget', 'arm'])[fields].mean().reset_index()
    yearly.to_csv(ROOT / 'year_budget_arm_means.csv', index=False)
    macro.to_csv(ROOT / 'macro_budget_arm_means.csv', index=False)
    cohorts(ROOT, records, {32, selected})
    global_metrics=pd.read_csv(ROOT.parent/'global/test_metrics.csv')
    global_ua=global_metrics[(global_metrics.arm=='UA') & global_metrics.seed.isin(seeds) & global_metrics.year.isin(years)]
    selected_ua=metrics[(metrics.budget==selected) & (metrics.arm=='UA') & metrics.seed.isin(seeds)]
    global_point={'global_UA_recall':float(global_ua.recall.mean()),'selected_distributed_UA_recall':float(selected_ua.recall.mean()),'distributed_minus_global_UA_difference_pp':100*float(selected_ua.recall.mean()-global_ua.recall.mean()),'same_seeds':seeds,'confidence_interval_computed':False,'capacity_not_selected_using_this_comparison':True}
    result = {'runs': len(records), 'selected_budget': selected, 'analysis_seeds': seeds,
              'capacity_selection': selection, 'comparisons': comparisons, 'same_information_Global_UA_point_reference':global_point,
              'seed_counts_by_capacity':{str(b):len({r['seed'] for r in records if r['budget']==b}) for b in (32,64,128)},
              'year_budget_arm_means': yearly.to_dict('records'), 'macro_budget_arm_means': macro.to_dict('records'),
              'nonselected_capacity_results': 'pilot only; never replace the fold1-selected capacity using fold3',
              'training_seconds_sum': sum(r['train_seconds'] for r in records),
              'payload_bytes': {str(b): b * 4 for b in (32, 64, 128)},
              'global_gain_used_as_gate': False, 'pilot_gate_sha256': gate_before,
              'capacity_selection_sha256': selection_before,
              'interpretation': 'current-snapshot audit source on reused development years; no independent temporal validation'}
    assert hashlib.sha256((ROOT / 'pilot_gate.json').read_bytes()).hexdigest() == gate_before
    assert hashlib.sha256((ROOT / 'capacity_selection.json').read_bytes()).hexdigest() == selection_before
    write_json(ROOT / 'summary.json', result)
    print(json.dumps({'runs': result['runs'], 'selected_budget': selected,
        'comparisons': {name: {'difference_pp': c['macro_mean_difference_pp'],
            'CI_pp': [c['company_cluster_bootstrap']['percentile_95_lower_pp'], c['company_cluster_bootstrap']['percentile_95_upper_pp']],
            'stable_benefit': c['stable_benefit_rule']['passed']} for name, c in comparisons.items()}},default=lambda x:x.item() if isinstance(x,np.generic) else str(x)), flush=True)


def append_secondary_same_information_capacity():
    assert (ROOT/'verification.json').exists()
    path=ROOT/'summary.json'
    result=json.loads(path.read_text())
    comparison=compare(ROOT,(result['selected_budget'],'UA'),(32,'UA'),[2020,2021],result['analysis_seeds'])
    comparison.update(role='Secondary same-information capacity comparison, requested after primary results; never used to select capacity or change the pilot gate',capacity_selection_changed=False)
    result['secondary_same_information_capacity_comparison']=comparison
    write_json(path,result)
    print(json.dumps({'secondary_comparison':'selected UA minus D32 UA','difference_pp':comparison['macro_mean_difference_pp'],'year_differences':comparison['year_mean_differences'],'CI_pp':[comparison['company_cluster_bootstrap']['percentile_95_lower_pp'],comparison['company_cluster_bootstrap']['percentile_95_upper_pp']]},default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),flush=True)

if __name__ == '__main__':
    import sys
    if '--secondary-only' not in sys.argv:
        main()
    append_secondary_same_information_capacity()
