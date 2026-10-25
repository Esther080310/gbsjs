"""Evaluate CMA against UA7 (non-inferiority) + control-code interpretability + counterfactual."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / '实验与代码/评价约束/20261018-v1'))
import contract as K  # noqa: E402
import cma_lib as C  # noqa: E402

FROZEN = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-evidence-v1/audit_information/distributed/D128'
REPEATS, BOOT = 2000, 20261018
CRITERIA = ('c1_non_inferior_0p5pp', 'c2_ci_lower_above_minus1p2', 'c3_each_year_above_minus1p0',
            'c4_ap_above_minus1p0', 'c5_budget_le_128', 'c6_control_code_separates_categories')


def target(year):
    pack = C.pack_for(year, range(4))
    s = pack['splits'][3]
    return s['y'].astype(int), s['keys'].Symbol.to_numpy(), s


def load(arm, year, seed):
    if arm == 'UA7':
        with np.load(FROZEN / str(year) / f'UA_s{seed}' / 'predictions.npz', allow_pickle=False) as z:
            return z['a_logit'].astype(float)
    with np.load(HERE / 'artifacts' / str(year) / f'CMA_s{seed}' / 'fold3.npz', allow_pickle=False) as z:
        return z['a_logit'].astype(float)


def rp(y, s):
    k = int(np.ceil(.1 * len(y)))
    return float(y[np.argsort(-s, kind='stable')[:k]].sum() / y.sum())


def curve(y, s, w):
    o = np.argsort(-s, kind='stable'); ww = w[:, o]; yy = y[o].astype(float)
    pos = ww @ yy; b = np.ceil(.1 * ww.sum(1)); before = np.cumsum(ww, axis=1) - ww
    sel = np.minimum(ww, np.maximum(b[:, None] - before, 0.0))
    return (sel @ yy) / pos


def main():
    syms = {y: target(y)[1] for y in C.YEARS}
    W = K.shared_company_weights(syms, REPEATS, BOOT)
    Y = {y: target(y)[0] for y in C.YEARS}
    per_year, per_seed = {}, {}
    for y in C.YEARS:
        d, sp = [], {}
        for s in C.SEEDS:
            d.append(curve(Y[y], load('CMA', y, s), W[y]) - curve(Y[y], load('UA7', y, s), W[y]))
            sp[str(s)] = 100 * (rp(Y[y], load('CMA', y, s)) - rp(Y[y], load('UA7', y, s)))
        per_year[y] = np.mean(d, axis=0); per_seed[y] = sp
    comb = np.mean([per_year[y] for y in C.YEARS], axis=0)
    exact = float(np.mean([np.mean([100 * (rp(Y[y], load('CMA', y, s)) - rp(Y[y], load('UA7', y, s)))
                                  for s in C.SEEDS]) for y in C.YEARS]))
    ap = float(np.mean([np.mean([average_precision_score(Y[y], load('CMA', y, s))
                                 - average_precision_score(Y[y], load('UA7', y, s))
                                 for s in C.SEEDS]) for y in C.YEARS]))
    # ---- interpretability: does the control code separate audit categories?
    formal = C.shared.load_loader().formal
    pack4 = C.pack_for(2020, range(4)); split = pack4['splits'][3]
    a = split['xs'][2][:, 16:23]
    # column 0 is the match flag (almost always 1), so the category is argmax over columns 1..6
    cats = np.argmax(a[:, 1:], axis=1) + 1
    print('2020 评价年审计类别分布:', {int(c): int((cats == c).sum()) for c in np.unique(cats)})
    with torch.no_grad():
        model = C.ControlMessageNet(formal, C.pack_for(2020), 42)
        sd = torch.load(HERE / 'artifacts/2020/CMA_s42/model.pt', map_location='cpu', weights_only=True)['state_dict']
        model.load_state_dict(sd)
        code = model.base.enc[2](torch.tensor(a, dtype=torch.float32)).numpy()
        gam = model.film(torch.tensor(code))[:, :88].numpy()
    within, between = [], []
    for c in np.unique(cats):
        idx = np.flatnonzero(cats == c)
        if len(idx) < 5:
            continue
        sub = code[idx]
        d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        within.append(np.median(d[np.triu_indices(len(idx), 1)]))
    for i, c1 in enumerate(np.unique(cats)):
        for c2 in np.unique(cats)[i + 1:]:
            i1, i2 = np.flatnonzero(cats == c1), np.flatnonzero(cats == c2)
            if len(i1) < 5 or len(i2) < 5:
                continue
            d = np.linalg.norm(code[i1][:, None, :] - code[i2][None, :, :], axis=-1)
            between.append(np.median(d))
    sep = float(np.median(between) / max(np.median(within), 1e-9)) if between and within else 0.0
    gamma_norm = {str(int(c)): float(np.linalg.norm(gam[cats == c], axis=1).mean())
                  for c in np.unique(cats) if (cats == c).sum() >= 5}
    checks = dict(c1_non_inferior_0p5pp=bool(exact >= -0.5),
                  c2_ci_lower_above_minus1p2=bool(100 * np.quantile(comb, .025) > -1.2),
                  c3_each_year_above_minus1p0=bool(all(100 * np.mean(per_year[y]) >= -1.0 for y in C.YEARS)),
                  c4_ap_above_minus1p0=bool(100 * ap >= -1.0),
                  c5_budget_le_128=bool(C.BUDGET <= 128),
                  c6_control_code_separates_categories=bool(sep > 1.0))
    K.assert_criteria_complete(CRITERIA, checks)
    K.assert_point_is_exact(exact, exact, 100 * float(np.mean(comb)), tol=1e-9)
    out = dict(exact_point_pp=exact, lower_pp=100 * float(np.quantile(comb, .025)),
               upper_pp=100 * float(np.quantile(comb, .975)),
               per_year_pp={str(y): 100 * float(np.mean(per_year[y])) for y in C.YEARS},
               per_seed_two_year_pp={s: float(np.mean([per_seed[y][str(s)] for y in C.YEARS]))
                                     for s in C.SEEDS},
               ap_diff_pp=100 * ap,
               budget=dict(cma=C.BUDGET, ua7=128, saving_pct=100 * (1 - C.BUDGET / 128)),
               interpretability=dict(separation_ratio=sep, within_median=float(np.median(within)),
                                     between_median=float(np.median(between)),
                                     gamma_norm_by_category=gamma_norm),
               checks=checks, adopt=all(checks.values()))
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'evaluation.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf8')
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()
                      if k in ('exact_point_pp', 'lower_pp', 'upper_pp', 'per_year_pp', 'ap_diff_pp',
                               'budget', 'interpretability', 'checks', 'adopt')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
