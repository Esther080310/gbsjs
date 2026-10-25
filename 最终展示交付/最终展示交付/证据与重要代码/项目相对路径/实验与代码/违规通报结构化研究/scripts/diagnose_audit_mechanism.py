"""Why does the audit-opinion category add ranking information? Mechanism diagnostic.

Zero new neural training: reuses the frozen six-seed predictions of the audit round and
the 2022 replication, plus the frozen audit features and caches.

Five questions, each answered out of sample (fit on fold2, evaluate on fold3, so nothing
is fitted on the evaluation year):

  A non-redundancy : given a flexible function of the *entire* baseline score (decile
                     bins, which absorb any monotone re-scaling), does the audit category
                     still add ranking information?
  B gradation      : is the signal just "modified opinion or not", or does the five-way
                     gradation carry more?
  C substitution   : is the audit gain concentrated where financial data is thin?
  D mechanism match: is the gain concentrated on accounting/financial violation types
                     rather than spread uniformly?
  E description    : what do the newly caught positives look like?

Company-clustered paired bootstrap is used for the AUC differences.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P / 'scripts'))
sys.path.insert(0, str(E / '发生年度实验'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from common import TYPE_CODES

AUDIT = P / 'artifacts/experiments/20261016-evidence-v1/audit_information'
REPL = P / 'artifacts/experiments/20261016-audit-year2022-v1'
OUT = P / 'artifacts/experiments/20261016-audit-mechanism-v1'
CACHE = E / '财务年度配对修正/cache'
SEEDS = (42, 100, 17, 223, 483, 907)
N_BOOT = 1000
RNG = np.random.default_rng(20261016)

TYPE_GROUP = {}
for code in TYPE_CODES:
    if code in ('P2501', 'P2503', 'P2515'):
        TYPE_GROUP[code] = '财务报告类'
    elif code in ('P2504', 'P2505', 'P2506'):
        TYPE_GROUP[code] = '披露类'
    elif code in ('P2509', 'P2510', 'P2514'):
        TYPE_GROUP[code] = '资金/担保类'
    else:
        TYPE_GROUP[code] = '其他'


def paths(year):
    if year == 2022:
        return REPL / 'features' / f'audit_features_{year}.npz', REPL / 'cache' / f'{year}.npz', 'repl'
    return AUDIT / 'features' / f'audit_features_{year}.npz', CACHE / f'{year}.npz', 'audit'


def predictions(year, budget, arm, seed, layout):
    if layout == 'repl':
        path = REPL / 'distributed' / f'D{budget}' / f'{arm}_s{seed}' / 'predictions.npz'
    else:
        path = AUDIT / 'distributed' / f'D{budget}' / str(year) / f'{arm}_s{seed}' / 'predictions.npz'
    with np.load(path, allow_pickle=False) as saved:
        return saved['3_a_logit'].copy()


def deciles(fit_scores):
    edges = np.quantile(fit_scores, np.linspace(0, 1, 11)[1:-1])
    return edges


def binned(scores, edges):
    index = np.digitize(scores, edges)
    out = np.zeros((len(scores), len(edges) + 1))
    out[np.arange(len(scores)), index] = 1.0
    return out


def fit_predict(x_fit, y_fit, x_test, C=1.0):
    model = LogisticRegression(C=C, max_iter=5000, class_weight='balanced')
    model.fit(x_fit, y_fit)
    return model.predict_proba(x_test)[:, 1]


def paired_auc_difference(y, score_a, score_b, companies, repeats=N_BOOT):
    unique = np.unique(companies)
    index = {c: i for i, c in enumerate(unique)}
    row = np.array([index[c] for c in companies])
    observed = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)
    draws = np.empty(repeats)
    for repeat in range(repeats):
        counts = RNG.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        weights = counts[row]
        picked = np.repeat(np.arange(len(y)), weights)
        if len(np.unique(y[picked])) < 2:
            draws[repeat] = np.nan
            continue
        draws[repeat] = roc_auc_score(y[picked], score_a[picked]) - roc_auc_score(y[picked], score_b[picked])
    draws = draws[~np.isnan(draws)]
    return observed, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def audit_columns(feature, fold, variant='UA'):
    return feature[f'{fold}_{variant}'].astype(np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_a, rows_b, rows_c, rows_d, rows_e = [], [], [], [], []
    for year in (2020, 2021, 2022):
        feature_path, cache_path, layout = paths(year)
        with np.load(feature_path, allow_pickle=False) as feature:
            y2 = feature['2_y'].astype(int); y3 = feature['3_y'].astype(int)
            ua2 = audit_columns(feature, 2); ua3 = audit_columns(feature, 3)
            r2 = feature['2_UM'].astype(np.float64)[:, :1]; r3 = feature['3_UM'].astype(np.float64)[:, :1]
            type3 = feature['3_types']; sym3 = feature['3_symbol'].astype(str)
        with np.load(cache_path, allow_pickle=False) as cache:
            q2 = cache['2_q'][:, 1].astype(np.float64); q3 = cache['3_q'][:, 1].astype(np.float64)
        baseline = {b: np.mean([predictions(year, b, 'U0', s, layout) for s in SEEDS], axis=0)
                    for b in (32, 128)}
        audit_model = np.mean([predictions(year, 128, 'UA', s, layout) for s in SEEDS], axis=0)

        for budget in (32, 128):
            s3 = baseline[budget]
            s2 = np.mean([_fold2_logits(year, budget, 'U0', s, layout) for s in SEEDS], axis=0)
            edges = deciles(s2)
            base_fit = binned(s2, edges)
            base_test = binned(s3, edges)
            base_only = fit_predict(base_fit, y2, base_test)
            base_plus = fit_predict(np.hstack([base_fit, ua2]), y2, np.hstack([base_test, ua3]))
            audit_only = fit_predict(ua2, y2, ua3)
            difference, low, high = paired_auc_difference(y3, base_plus, base_only, sym3)
            rows_a.append({'year': year, 'baseline': f'D{budget}_U0_seedmean', 'n_test': len(y3),
                           'positives': int(y3.sum()),
                           'auc_base': roc_auc_score(y3, base_only),
                           'auc_base_plus_audit': roc_auc_score(y3, base_plus),
                           'auc_audit_only': roc_auc_score(y3, audit_only),
                           'delta_auc': difference, 'delta_auc_lo': low, 'delta_auc_hi': high,
                           'logloss_base': log_loss(y3, base_only),
                           'logloss_base_plus_audit': log_loss(y3, base_plus),
                           'logloss_audit_only': log_loss(y3, audit_only),
                           'fold3_model_auc': roc_auc_score(y3, audit_model)})
        # B gradation. ua columns are [r, c1_standard, c2..c5 modified, c6_unknown], so the
        # binary "modified opinion" flag must exclude column 1 (standard unqualified).
        edges = deciles(np.mean([_fold2_logits(year, 128, 'U0', s, layout) for s in SEEDS], axis=0))
        nonstandard2 = ua2[:, 2:].sum(1, keepdims=True)
        nonstandard3 = ua3[:, 2:].sum(1, keepdims=True)
        for name, fit_cols, test_cols, dims in (
                ('r_only', r2, r3, 1),
                ('r+nonstandard_binary', np.hstack([r2, nonstandard2]), np.hstack([r3, nonstandard3]), 2),
                ('r+5categories', ua2, ua3, 6)):
            probability = fit_predict(fit_cols, y2, test_cols)
            rows_b.append({'year': year, 'audit_model': name, 'columns': dims,
                           'auc': roc_auc_score(y3, probability), 'logloss': log_loss(y3, probability),
                           'fold3_nonstandard_base_rate': float(nonstandard3.mean())})
        # C substitution by financial observation rate
        strata = [('财务全空 q2=0', q3 == 0), ('0<q2<0.5', (q3 > 0) & (q3 < .5)),
                  ('0.5<=q2<0.9', (q3 >= .5) & (q3 < .9)), ('q2>=0.9', q3 >= .9)]
        s2 = np.mean([_fold2_logits(year, 128, 'U0', s, layout) for s in SEEDS], axis=0)
        edges = deciles(s2)
        for name, mask in strata:
            if mask.sum() < 50 or y3[mask].sum() < 5:
                rows_c.append({'year': year, 'stratum': name, 'n': int(mask.sum()),
                               'positives': int(y3[mask].sum()), 'note': '支持量不足'})
                continue
            base_only = fit_predict(binned(s2, edges), y2, binned(baseline[128], edges)[mask])
            base_plus = fit_predict(np.hstack([binned(s2, edges), ua2]), y2,
                                    np.hstack([binned(baseline[128], edges), ua3])[mask])
            order_base = np.argsort(-baseline[128][mask], kind='stable')
            order_audit = np.argsort(-audit_model[mask], kind='stable')
            k = math.ceil(.1 * int(mask.sum()))
            recall_base = y3[mask][order_base[:k]].sum() / max(y3[mask].sum(), 1)
            recall_audit = y3[mask][order_audit[:k]].sum() / max(y3[mask].sum(), 1)
            rows_c.append({'year': year, 'stratum': name, 'n': int(mask.sum()),
                           'positives': int(y3[mask].sum()),
                           'auc_base': roc_auc_score(y3[mask], base_only),
                           'auc_base_plus_audit': roc_auc_score(y3[mask], base_plus),
                           'recall_base_D128_U0': recall_base, 'recall_D128_UA': recall_audit,
                           'recall_delta_pp': 100 * (recall_audit - recall_base)})
        # D mechanism match by violation type group
        order_base = np.argsort(-baseline[32], kind='stable')
        order_audit = np.argsort(-audit_model, kind='stable')
        k = math.ceil(.1 * len(y3))
        for group in ('财务报告类', '披露类', '资金/担保类', '其他'):
            columns = [j for j, code in enumerate(TYPE_CODES) if TYPE_GROUP[code] == group]
            positive_in_group = (type3[:, columns].sum(1) > 0) & (y3 == 1)
            support = int(positive_in_group.sum())
            if support < 10:
                rows_d.append({'year': year, 'type_group': group, 'positives': support,
                               'note': '支持量不足'})
                continue
            hits_base = int(positive_in_group[order_base[:k]].sum())
            hits_audit = int(positive_in_group[order_audit[:k]].sum())
            rows_d.append({'year': year, 'type_group': group, 'positives': support,
                           'caught_base': hits_base, 'caught_audit': hits_audit,
                           'delta_pp': 100 * (hits_audit - hits_base) / support})
        # E description of newly caught positives
        selected_base = np.zeros(len(y3), dtype=bool); selected_base[order_base[:k]] = True
        selected_audit = np.zeros(len(y3), dtype=bool); selected_audit[order_audit[:k]] = True
        added = selected_audit & ~selected_base & (y3 == 1)
        lost = selected_base & ~selected_audit & (y3 == 1)
        for label, mask in (('newly_caught', added), ('no_longer_caught', lost)):
            if mask.sum() == 0:
                continue
            matched = ua3[mask][:, 0] == 1
            rows_e.append({'year': year, 'kind': label, 'count': int(mask.sum()),
                           'matched_share': float(matched.mean()),
                           'standard_share_of_matched': float((ua3[mask][matched][:, 1] == 1).mean()) if matched.any() else None,
                           'nonstandard_share_of_matched': float((ua3[mask][matched][:, 2:].sum(1) == 1).mean()) if matched.any() else None,
                           'fold3_nonstandard_base_rate': float((ua3[:, 2:].sum(1) == 1).mean()),
                           'median_q2': float(np.median(q3[mask])),
                           'financial_reporting_share': float(
                               (type3[mask][:, [j for j, c in enumerate(TYPE_CODES) if TYPE_GROUP[c] == '财务报告类']].sum(1) > 0).mean())})
        print(f'--- {year} done', flush=True)
    for name, rows in (('A_non_redundancy', rows_a), ('B_gradation', rows_b),
                       ('C_substitution', rows_c), ('D_mechanism_match', rows_d), ('E_description', rows_e)):
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
        print(f'\n=== {name} ===')
        print(pd.DataFrame(rows).to_string(index=False))
    (OUT / 'diagnostic.json').write_text(json.dumps({'years': [2020, 2021, 2022], 'seeds': list(SEEDS),
                                                     'fit_fold': 2, 'evaluate_fold': 3,
                                                     'bootstrap': N_BOOT}, ensure_ascii=False, indent=2) + '\n',
                                        encoding='utf-8')


def _fold2_logits(year, budget, arm, seed, layout):
    if layout == 'repl':
        path = REPL / 'distributed' / f'D{budget}' / f'{arm}_s{seed}' / 'predictions.npz'
    else:
        path = AUDIT / 'distributed' / f'D{budget}' / str(year) / f'{arm}_s{seed}' / 'predictions.npz'
    with np.load(path, allow_pickle=False) as saved:
        return saved['2_a_logit'].copy()


if __name__ == '__main__':
    main()
