"""Do LLM-extracted violation mechanisms occupy distinguishable regions of X?

Two tests on positive company-years, each with a stratified permutation null so that
economic year and industry cannot by themselves produce apparent separability:

  T1 distance  : between-mechanism vs within-mechanism dispersion (PERMANOVA-style
                 pseudo-F on standardised principal components).
  T2 predictive: grouped cross-validated AUC for "mechanism k vs other positives",
                 L2 logistic regression on the same components, grouped by company so a
                 firm cannot be in both train and test.

Both run on (a) the model-free transformed inputs and (b) the frozen S32 32-dim message
representation, so a representation bottleneck can be told apart from a head bottleneck.

Control: a year+industry-only classifier, to show how much of any apparent signal is just
time and industry rather than mechanism-specific financial signature.

Offline: no new model training, no API.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P / 'scripts'))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

OUT = P / 'artifacts/experiments/20261016-mechanism-separability-v1'
CACHE = E / '财务年度配对修正/cache'
MODEL_ROOT = E / '财务年度配对修正/results'
PRIMARY = ('disclosure_omission', 'disclosure_delay', 'financial_misstatement',
           'related_party_transaction')
N_PERM = 1000
N_PERM_AUC = 200
N_PC = 10
RNG = np.random.default_rng(20261016)


def build_index():
    index = {}
    for cache_year in (2021, 2020):
        with np.load(CACHE / f'{cache_year}.npz', allow_pickle=False) as cache:
            for fold in range(4):
                for row, (s, y) in enumerate(zip(cache[f'{fold}_symbol'].astype(str),
                                                 cache[f'{fold}_year'])):
                    index.setdefault((s, int(y)), (cache_year, fold, row))
    return index


def _pack(cache_year, fold):
    meta = json.loads((CACHE / f'{cache_year}.json').read_text())
    with np.load(CACHE / f'{cache_year}.npz', allow_pickle=False) as cache:
        split = {'xs': [cache[f'{fold}_x{i}'].copy() for i in range(3)],
                 **{k: cache[f'{fold}_{k}'].copy() for k in ('industry', 'q', 'state', 'y', 'types')}}
        split['keys'] = pd.DataFrame({'Symbol': cache[f'{fold}_symbol'].astype(str),
                                      'economic_year': cache[f'{fold}_year']})
    return dict(meta, splits=[split])


def _message(formal, model, pack, row):
    split = pack['splits'][0]
    small = {'xs': [x[row:row + 1] for x in split['xs']],
             **{k: split[k][row:row + 1] for k in ('industry', 'q', 'state', 'y', 'types')}}
    model.eval()
    with torch.no_grad():
        model(*formal.prev.batch(formal.prev.tensor_split(small, 'cpu'), np.array([0])))
    return np.concatenate([m.numpy().ravel() for m in model.messages])


def load_features(units, index, representation):
    rows, keep = [], []
    formal, models = None, {}
    if representation == 'message':
        import importlib.util
        spec = importlib.util.spec_from_file_location('corrected_loader', E / '财务年度配对修正/pipeline.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        formal = loader.formal
    for position, unit in enumerate(units.itertuples()):
        key = (unit.symbol, unit.economic_year)
        if key not in index:
            continue
        cache_year, fold, row = index[key]
        with np.load(CACHE / f'{cache_year}.npz', allow_pickle=False) as cache:
            industry = int(cache[f'{fold}_industry'][row])
            if representation == 'raw':
                vector = np.concatenate([cache[f'{fold}_x{i}'][row] for i in range(3)])
            else:
                if (cache_year, fold) not in models:
                    pack = _pack(cache_year, fold)
                    model = formal.RiskModel(pack, 32, 'Plain', 42)
                    model.load_state_dict(torch.load(MODEL_ROOT / str(cache_year) / 'S32_s42' / 'model.pt',
                                                     weights_only=True)['state_dict'])
                    models[cache_year, fold] = (model, pack)
                model, pack = models[cache_year, fold]
                vector = _message(formal, model, pack, row)
        rows.append(vector)
        keep.append({'symbol': unit.symbol, 'economic_year': unit.economic_year,
                     'dominant': unit.dominant, 'industry': industry, 'cache_year': cache_year})
    return np.asarray(rows, dtype=np.float64), pd.DataFrame(keep)


def stat_with(values, labels):
    grand = values.mean(0)
    between = within = 0.0
    for group in np.unique(labels):
        members = values[labels == group]
        if len(members) < 2:
            continue
        centroid = members.mean(0)
        between += len(members) * float(((centroid - grand) ** 2).sum())
        within += float(((members - centroid) ** 2).sum())
    k = len(np.unique(labels))
    return (between / max(k - 1, 1)) / max(within / max(len(values) - k, 1), 1e-12)


def permute_within(binary, strata, rng):
    out = binary.copy()
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        out[members] = rng.permutation(out[members])
    return out


def cv_auc(components, binary, groups):
    if len(np.unique(binary)) < 2:
        return float('nan')
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261016)
    scores = np.full(len(binary), np.nan)
    for train, test in folds.split(components, binary, groups):
        if len(np.unique(binary[train])) < 2:
            return float('nan')
        model = LogisticRegression(penalty='l2', C=1.0, max_iter=5000, class_weight='balanced')
        model.fit(components[train], binary[train])
        scores[test] = model.decision_function(components[test])
    return float(roc_auc_score(binary, scores))


def components_of(features):
    scaler = StandardScaler().fit(features)
    return PCA(n_components=min(N_PC, features.shape[1], len(features) - 1),
               random_state=0).fit_transform(scaler.transform(features))


def main():
    units = pd.read_csv(OUT / 'mechanism_company_years.csv', dtype={'symbol': str})
    units['symbol'] = units.symbol.str.zfill(6)
    units = units.drop_duplicates(subset=['symbol', 'economic_year'])
    units = units[units.y == 1].reset_index(drop=True)
    index = build_index()
    mapped = sum((u.symbol, u.economic_year) in index for u in units.itertuples())
    print(f'unique positive company-years with mechanism labels: {len(units)}; mapped: {mapped}')

    results = {}
    aligned = {}
    for representation in ('raw', 'message'):
        matrix, meta = load_features(units, index, representation)
        mask = meta.dominant.isin(PRIMARY).to_numpy()
        features = matrix[mask]
        labels = meta.dominant.to_numpy()[mask]
        strata = np.array([f'{y}|{i}' for y, i in zip(meta.economic_year[mask], meta.industry[mask])])
        groups = meta.symbol.to_numpy()[mask]
        components = components_of(features)
        observed = stat_with(components, labels)
        null = np.array([stat_with(components, permute_within(labels, strata, RNG)) for _ in range(N_PERM)])
        p_value = float((null >= observed).mean())
        per_mechanism = {}
        for mechanism in PRIMARY:
            binary = (labels == mechanism).astype(int)
            auc = cv_auc(components, binary, groups)
            null_auc = np.array([cv_auc(components, permute_within(binary, strata, RNG), groups)
                                 for _ in range(N_PERM_AUC)])
            null_auc = null_auc[~np.isnan(null_auc)]
            per_mechanism[mechanism] = {
                'n': int(binary.sum()), 'cv_auc': auc,
                'null_median': float(np.median(null_auc)) if len(null_auc) else None,
                'null_p95': float(np.quantile(null_auc, .95)) if len(null_auc) else None,
                'permutation_p': float((null_auc >= auc).mean()) if len(null_auc) else None}
        results[representation] = {'n': int(mask.sum()), 'dimensions': int(features.shape[1]),
                                   'pseudo_F': float(observed), 'pseudo_F_permutation_p': p_value,
                                   'per_mechanism': per_mechanism}
        aligned[representation] = meta
        print(f'\n=== {representation} ({features.shape[1]} dims, n={int(mask.sum())}, '
              f'years {meta.economic_year[mask].min()}-{meta.economic_year[mask].max()}) ===')
        print(f'  T1 pseudo-F = {observed:.3f}   stratified permutation p = {p_value:.3f}')
        for mechanism, row in per_mechanism.items():
            print(f"  T2 {mechanism:26s} n={row['n']:3d}  CV-AUC={row['cv_auc']:.3f}  "
                  f"null median={row['null_median']:.3f} p95={row['null_p95']:.3f}  "
                  f"p={row['permutation_p']:.3f}")

    print('\n=== control: year + industry only (S32 representation) ===')
    meta = aligned['message']
    mask = meta.dominant.isin(PRIMARY).to_numpy()
    labels = meta.dominant.to_numpy()[mask]
    dummies = pd.get_dummies(pd.DataFrame({'y': meta.economic_year.to_numpy()[mask],
                                           'i': meta.industry.to_numpy()[mask]})).to_numpy(float)
    control = components_of(dummies)
    control_result = {}
    for mechanism in PRIMARY:
        binary = (labels == mechanism).astype(int)
        auc = cv_auc(control, binary, meta.symbol.to_numpy()[mask])
        control_result[mechanism] = {'n': int(binary.sum()), 'cv_auc_year_industry': auc}
        print(f"  {mechanism:26s} n={int(binary.sum()):3d}  CV-AUC(year+industry)={auc:.3f}")
    results['year_industry_control'] = control_result
    (OUT / 'separability_results.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\nwritten', OUT / 'separability_results.json')


if __name__ == '__main__':
    main()
