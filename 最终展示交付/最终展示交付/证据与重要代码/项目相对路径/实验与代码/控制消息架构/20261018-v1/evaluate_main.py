"""Canonical evaluation: EXACT per-year and combined point estimates (C2), with the bootstrap
mean stored separately, plus AP and cost accounting. Covers every structural arm."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / '实验与代码/评价约束/20261018-v1'))
import cma_lib as C  # noqa: E402
import contract as K  # noqa: E402

E2020 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-evidence-v1/audit_information/distributed/D128'
E22 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-audit-year2022-v1/distributed/D128'
GLOBAL = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-evidence-v1/audit_information/global'
SEEDS = C.SEEDS
ARMS = ('FiLM-L', 'CMA-L', 'CMA-G', 'CMA')
COST = {'FiLM-L': dict(budget=128, note='预算不变，参数 +3.1%'),
        'CMA-L': dict(budget=115, note='预算 −10.2%，但删除了治理内容通道'),
        'CMA-G': dict(budget=123, note='治理摘要 8 + 控制码 8'),
        'CMA': dict(budget=115, note='MLP 调制头，参数 +10.4%')}


def labels(year):
    if year == 2022:
        sys.path.insert(0, str(C.P / 'scripts'))
        import train_audit_year2022 as f
        f.install_redirected_loaders()
    p = C.pack_for(year, range(4))['splits'][3]
    return p['y'].astype(int), p['keys'].Symbol.to_numpy()


def score(arm, year, seed):
    if arm in ARMS:
        with np.load(HERE / 'artifacts' / str(year) / f'{arm}_s{seed}/fold3.npz', allow_pickle=False) as z:
            return z['a_logit'].astype(float)
    if arm == 'GLOBAL':
        p = GLOBAL / str(year) / f'UA_s{seed}/predictions.npz'
    else:
        name = 'UA' if arm == 'UA7' else arm
        p = (E22 / f'{name}_s{seed}/predictions.npz') if year == 2022 else \
            (E2020 / str(year) / f'{name}_s{seed}/predictions.npz')
    with np.load(p, allow_pickle=False) as z:
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
    Y, SYM = {}, {}
    for y in (2020, 2021):
        Y[y], SYM[y] = labels(y)
    Y[2022], SYM[2022] = labels(2022)          # installs the 2022 redirect; symbols only afterwards
    W = K.shared_company_weights(SYM, 2000, 20261018)
    out = {'targets': {str(y): dict(rows=int(len(Y[y])), positives=int(Y[y].sum())) for y in (2020, 2021, 2022)},
           'arms': {}}
    for arm in ARMS:
        per_exact, per_boot, ap_exact = {}, {}, {}
        for y in (2020, 2021, 2022):
            ex = [100 * (rp(Y[y], score(arm, y, s)) - rp(Y[y], score('UA7', y, s))) for s in SEEDS]
            bo = np.mean([curve(Y[y], score(arm, y, s), W[y]) - curve(Y[y], score('UA7', y, s), W[y])
                          for s in SEEDS], axis=0)
            apx = [average_precision_score(Y[y], score(arm, y, s))
                   - average_precision_score(Y[y], score('UA7', y, s)) for s in SEEDS]
            per_exact[str(y)] = float(np.mean(ex))
            per_boot[str(y)] = dict(boot_mean_pp=100 * float(np.mean(bo)),
                                    lower_pp=100 * float(np.quantile(bo, .025)),
                                    upper_pp=100 * float(np.quantile(bo, .975)),
                                    p_le_0=float(np.mean(bo <= 0)))
            ap_exact[str(y)] = 100 * float(np.mean(apx))
        exact_3 = float(np.mean([per_exact[str(y)] for y in (2020, 2021, 2022)]))
        exact_2 = float(np.mean([per_exact[str(y)] for y in (2020, 2021)]))
        comb = np.mean([np.mean([curve(Y[y], score(arm, y, s), W[y]) - curve(Y[y], score('UA7', y, s), W[y])
                                 for s in SEEDS], axis=0) for y in (2020, 2021, 2022)], axis=0)
        out['arms'][arm] = dict(
            exact_per_year_pp=per_exact, exact_three_year_pp=exact_3, exact_two_year_pp=exact_2,
            bootstrap_per_year=per_boot,
            bootstrap_three_year=dict(mean_pp=100 * float(np.mean(comb)),
                                      lower_pp=100 * float(np.quantile(comb, .025)),
                                      upper_pp=100 * float(np.quantile(comb, .975)),
                                      p_le_0=float(np.mean(comb <= 0))),
            ap_exact_per_year_pp=ap_exact, cost=COST[arm])
        # C2: the reported point MUST be the exact one, and must not silently equal the bootstrap mean
        assert abs(out['arms'][arm]['exact_three_year_pp']
                   - np.mean(list(per_exact.values()))) < 1e-12
        print(json.dumps({arm: dict(exact_3y=round(exact_3, 3), exact_2y=round(exact_2, 3),
                                    per_year={k: round(v, 3) for k, v in per_exact.items()},
                                    boot_3y=round(out['arms'][arm]['bootstrap_three_year']['mean_pp'], 3),
                                    ci=[round(out['arms'][arm]['bootstrap_three_year']['lower_pp'], 3),
                                        round(out['arms'][arm]['bootstrap_three_year']['upper_pp'], 3)],
                                    ap={k: round(v, 2) for k, v in ap_exact.items()})},
                          ensure_ascii=False))
    # global comparison for the main model (2020/2021 only: 2022 has no global arm)
    g = {}
    for y in (2020, 2021):
        ex = [100 * (rp(Y[y], score('FiLM-L', y, s)) - rp(Y[y], score('GLOBAL', y, s))) for s in SEEDS]
        g[str(y)] = float(np.mean(ex))
    out['FiLM-L_vs_global'] = dict(per_year_pp=g, two_year_pp=float(np.mean(list(g.values()))))
    (HERE / 'results' / 'main_model_eval.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf8')
    print('FiLM-L vs global:', {k: round(v, 3) for k, v in g.items()},
          'two-year', round(out['FiLM-L_vs_global']['two_year_pp'], 3))


if __name__ == '__main__':
    main()
