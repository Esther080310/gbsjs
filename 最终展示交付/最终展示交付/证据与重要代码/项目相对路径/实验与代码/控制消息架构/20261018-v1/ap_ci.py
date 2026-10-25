"""Company-clustered bootstrap for the AP difference (arm - UA7)."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import cma_lib as C

E2020 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-evidence-v1/audit_information/distributed/D128'
E22 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-audit-year2022-v1/distributed/D128'
SEEDS = C.SEEDS
DRAWS = 800


def labels(year):
    if year == 2022:
        sys.path.insert(0, str(C.P / 'scripts'))
        import train_audit_year2022 as f
        f.install_redirected_loaders()
    p = C.pack_for(year, range(4))['splits'][3]
    return p['y'].astype(int), p['keys'].Symbol.to_numpy()


def score(arm, year, seed):
    if arm != 'UA7':
        with np.load(HERE / 'artifacts' / str(year) / f'{arm}_s{seed}/fold3.npz', allow_pickle=False) as z:
            return z['a_logit'].astype(float)
    p = (E22 / f'UA_s{seed}/predictions.npz') if year == 2022 else \
        (E2020 / str(year) / f'UA_s{seed}/predictions.npz')
    with np.load(p, allow_pickle=False) as z:
        return z['a_logit'].astype(float)


out = {}
for year in (2020, 2021, 2022):
    y, sym = labels(year)
    import pandas as pd
    codes = pd.factorize(sym)[0]
    n = codes.max() + 1
    idx_by = [np.flatnonzero(codes == c) for c in range(n)]
    rng = np.random.default_rng(20261018)
    draws = [np.concatenate([idx_by[c] for c in rng.integers(0, n, n)]) for _ in range(DRAWS)]
    ref = {s: score('UA7', year, s) for s in SEEDS}
    for arm in ('CMA-G', 'FiLM-L', 'CMA-L'):
        sc = {s: score(arm, year, s) for s in SEEDS}
        exact = float(np.mean([average_precision_score(y, sc[s]) - average_precision_score(y, ref[s])
                               for s in SEEDS]))
        boot = []
        for ix in draws:
            yy = y[ix]
            if yy.sum() == 0:
                continue
            boot.append(np.mean([average_precision_score(yy, sc[s][ix]) - average_precision_score(yy, ref[s][ix])
                                 for s in SEEDS]))
        boot = np.array(boot)
        out.setdefault(arm, {})[str(year)] = dict(exact_pp=100 * exact,
                                                  lower_pp=100 * float(np.quantile(boot, .025)),
                                                  upper_pp=100 * float(np.quantile(boot, .975)),
                                                  p_le_0=float(np.mean(boot <= 0)))
    print(year, {a: {k: round(v, 2) for k, v in out[a][str(year)].items()} for a in out})
(HERE / 'results' / 'ap_ci.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
