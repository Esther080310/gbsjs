"""Exact-point comparison of CMA-G / FiLM-L / CMA-L against UA7 (three years where available)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import cma_lib as C  # noqa: E402

E2020 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-evidence-v1/audit_information/distributed/D128'
E22 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-audit-year2022-v1/distributed/D128'
SEEDS = C.SEEDS


def labels(year):
    if year == 2022:
        sys.path.insert(0, str(C.P / 'scripts'))
        import train_audit_year2022 as f
        f.install_redirected_loaders()
    return C.pack_for(year, range(4))['splits'][3]['y'].astype(int)


def score(arm, year, seed):
    if arm in ('CMA', 'CMA-L', 'CMA-G', 'FiLM-L'):
        p = HERE / 'artifacts' / str(year) / f'{arm}_s{seed}/fold3.npz'
        with np.load(p, allow_pickle=False) as z:
            return z['a_logit'].astype(float)
    name = 'UA' if arm == 'UA7' else arm
    p = (E22 / f'{name}_s{seed}/predictions.npz') if year == 2022 else \
        (E2020 / str(year) / f'{name}_s{seed}/predictions.npz')
    with np.load(p, allow_pickle=False) as z:
        return z['a_logit'].astype(float)


def rp(y, s):
    k = int(np.ceil(.1 * len(y)))
    return float(y[np.argsort(-s, kind='stable')[:k]].sum() / y.sum())


out = {}
for year in (2020, 2021):
    y = labels(year)
    for arm in ('CMA-G', 'FiLM-L', 'CMA-L'):
        d = [100 * (rp(y, score(arm, year, s)) - rp(y, score('UA7', year, s))) for s in SEEDS]
        ap = [average_precision_score(y, score(arm, year, s))
              - average_precision_score(y, score('UA7', year, s)) for s in SEEDS]
        out.setdefault(arm, {})[str(year)] = dict(exact_pp=float(np.mean(d)),
                                                  ap_pp=100 * float(np.mean(ap)))
    print(year, {a: round(out[a][str(year)]['exact_pp'], 3) for a in out})
# 2022 only for FiLM-L / CMA-L
y = labels(2022)
for arm in ('FiLM-L', 'CMA-L', 'CMA-G'):
    d = [100 * (rp(y, score(arm, 2022, s)) - rp(y, score('UA7', 2022, s))) for s in SEEDS]
    ap = [average_precision_score(y, score(arm, 2022, s))
          - average_precision_score(y, score('UA7', 2022, s)) for s in SEEDS]
    out.setdefault(arm, {})['2022'] = dict(exact_pp=float(np.mean(d)), ap_pp=100 * float(np.mean(ap)))
print('2022', {a: round(out[a]['2022']['exact_pp'], 3) for a in ('FiLM-L', 'CMA-L', 'CMA-G')})
for arm, yrs in out.items():
    vals = [yrs[k]['exact_pp'] for k in ('2020', '2021', '2022') if k in yrs]
    print(f'{arm}: 各年 {[round(v,3) for v in vals]} | 等权 {np.mean(vals):+.3f}pp | '
          f'AP {[round(yrs[k]["ap_pp"],3) for k in yrs]}')
(HERE / 'results' / 'ablation_eval.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                     encoding='utf8')
