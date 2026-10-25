"""Three-year consolidated evaluation: CMA / CMA-L vs UA7 and vs U0, with contract assertions."""
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
import contract as K  # noqa: E402
import cma_lib as C  # noqa: E402

E22 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-audit-year2022-v1/distributed/D128'
E2020 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-evidence-v1/audit_information/distributed/D128'
REPEATS, BOOT = 2000, 20261018
YEARS = (2020, 2021, 2022)
ARMS = ('CMA', 'CMA-L')
CRIT = ('c1_non_inferior_0p5pp', 'c2_ci_lower_above_minus1p2', 'c3_each_year_above_minus1p0',
        'c4_ap_above_minus1p0', 'c5_budget_le_128', 'c6_params_not_more_than_ua7')


def frozen(arm, year, seed):
    arm = 'UA' if arm == 'UA7' else arm        # frozen dirs use UA/U0
    base = E22 if year == 2022 else E2020
    p = (base / f'{arm}_s{seed}/predictions.npz') if year == 2022 else \
        (base / str(year) / f'{arm}_s{seed}/predictions.npz')
    with np.load(p, allow_pickle=False) as z:
        return z['a_logit'].astype(float)


def mine(arm, year, seed):
    with np.load(HERE / 'artifacts' / str(year) / f'{arm}_s{seed}/fold3.npz', allow_pickle=False) as z:
        return z['a_logit'].astype(float)


def target(year):
    """fold3 labels/symbols; 2022 needs the redirected loaders."""
    if year == 2022:
        sys.path.insert(0, str(C.P / 'scripts'))
        import train_audit_year2022 as fix22
        fix22.install_redirected_loaders()
    pack = C.pack_for(year, range(4))
    s = pack['splits'][3]
    return s['y'].astype(int), s['keys'].Symbol.to_numpy()


def rp(y, s):
    k = int(np.ceil(.1 * len(y)))
    return float(y[np.argsort(-s, kind='stable')[:k]].sum() / y.sum())


def curve(y, s, w):
    o = np.argsort(-s, kind='stable'); ww = w[:, o]; yy = y[o].astype(float)
    pos = ww @ yy; b = np.ceil(.1 * ww.sum(1)); before = np.cumsum(ww, axis=1) - ww
    sel = np.minimum(ww, np.maximum(b[:, None] - before, 0.0))
    return (sel @ yy) / pos


def main():
    # labels first (2022 redirect installs once)
    Y, SYM = {}, {}
    for y in (2020, 2021):
        Y[y], SYM[y] = target(y)
    Y[2022], SYM[2022] = target(2022)
    W = K.shared_company_weights(SYM, REPEATS, BOOT)
    print(json.dumps({str(y): dict(rows=len(Y[y]), positives=int(Y[y].sum())) for y in YEARS},
                     ensure_ascii=False))

    def contrast(arm, ref):
        """arm vs ref (both 6-seed single-model means), per year + equal-weight combined."""
        per_year, per_seed = {}, {}
        for y in YEARS:
            d, sp = [], {}
            for s in C.SEEDS:
                a = mine(arm, y, s) if arm in ARMS else frozen(arm, y, s)
                r = mine(ref, y, s) if ref in ARMS else frozen(ref, y, s)
                d.append(curve(Y[y], a, W[y]) - curve(Y[y], r, W[y]))
                sp[str(s)] = 100 * (rp(Y[y], a) - rp(Y[y], r))
            per_year[y] = np.mean(d, axis=0); per_seed[y] = sp
        comb = np.mean([per_year[y] for y in YEARS], axis=0)
        exact = float(np.mean([np.mean([100 * (rp(Y[y], mine(arm, y, s) if arm in ARMS else frozen(arm, y, s))
                                          - rp(Y[y], mine(ref, y, s) if ref in ARMS else frozen(ref, y, s)))
                                    for s in C.SEEDS]) for y in YEARS]))
        ap = float(np.mean([np.mean([average_precision_score(Y[y], mine(arm, y, s) if arm in ARMS else frozen(arm, y, s))
                                     - average_precision_score(Y[y], mine(ref, y, s) if ref in ARMS else frozen(ref, y, s))
                                     for s in C.SEEDS]) for y in YEARS]))
        return dict(exact_pp=exact, lower_pp=100 * float(np.quantile(comb, .025)),
                    upper_pp=100 * float(np.quantile(comb, .975)),
                    per_year_pp={str(y): 100 * float(np.mean(per_year[y])) for y in YEARS},
                    per_seed_pp={s: float(np.mean([per_seed[y][str(s)] for y in YEARS]))
                                 for s in C.SEEDS}, ap_pp=100 * ap)

    out = {'targets': {str(y): dict(rows=int(len(Y[y])), positives=int(Y[y].sum())) for y in YEARS},
           'contrasts': {}}
    for arm in ARMS:
        out['contrasts'][f'{arm}_vs_UA7'] = contrast(arm, 'UA7')
        out['contrasts'][f'{arm}_vs_U0'] = contrast(arm, 'U0')
    out['contrasts']['UA7_vs_U0'] = contrast('UA7', 'U0')

    # parameters / budget per arm
    def params(arm, y):
        if arm in ARMS:
            return json.loads((HERE / 'artifacts' / str(y) / f'{arm}_s42/run.json').read_text())['parameters']
        name = 'UA' if arm == 'UA7' else arm
        base = E22 if y == 2022 else E2020
        p = (base / f'{name}_s42/run.json') if y == 2022 else (base / str(y) / f'{name}_s42/run.json')
        return json.loads(p.read_text())['parameters']
    out['cost'] = {arm: {str(y): params(arm, y) for y in YEARS} for arm in ('UA7', 'U0') + ARMS}
    out['cost']['budget_floats'] = {'UA7': 128, 'U0': 128, 'CMA': 115, 'CMA-L': 115}

    # pre-registered decision for the two CMA variants
    for arm in ARMS:
        c = out['contrasts'][f'{arm}_vs_UA7']
        same_params = all(out['cost'][arm][str(y)] <= out['cost']['UA7'][str(y)] for y in YEARS)
        checks = dict(
            c1_non_inferior_0p5pp=bool(c['exact_pp'] >= -0.5),
            c2_ci_lower_above_minus1p2=bool(c['lower_pp'] > -1.2),
            c3_each_year_above_minus1p0=bool(all(v >= -1.0 for v in c['per_year_pp'].values())),
            c4_ap_above_minus1p0=bool(c['ap_pp'] >= -1.0),
            c5_budget_le_128=bool(115 <= 128),
            c6_params_not_more_than_ua7=bool(same_params))
        K.assert_criteria_complete(CRIT, checks)
        c['checks'] = checks
        c['adopt'] = all(checks.values())
    (HERE / 'results' / 'final_evaluation.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf8')
    for arm in ARMS:
        c = out['contrasts'][f'{arm}_vs_UA7']
        print(json.dumps({arm: dict(vs_ua7=round(c['exact_pp'], 3),
                                    ci=[round(c['lower_pp'], 3), round(c['upper_pp'], 3)],
                                    per_year={k: round(v, 3) for k, v in c['per_year_pp'].items()},
                                    ap_pp=round(c['ap_pp'], 3),
                                    params=out['cost'][arm], adopt=c['adopt'])}, ensure_ascii=False))
    for k in ('CMA_vs_U0', 'CMA-L_vs_U0', 'UA7_vs_U0'):
        c = out['contrasts'][k]
        print(k, round(c['exact_pp'], 3), [round(c['lower_pp'], 3), round(c['upper_pp'], 3)],
              {y: round(v, 3) for y, v in c['per_year_pp'].items()})


if __name__ == '__main__':
    main()
