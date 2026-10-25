"""Recall intervals (2-year and 3-year) for the three structural arms vs UA7."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / '实验与代码/评价约束/20261018-v1'))
import cma_lib as C, contract as K

E2020 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-evidence-v1/audit_information/distributed/D128'
E22 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-audit-year2022-v1/distributed/D128'
SEEDS = C.SEEDS


def curve(y, sc, w):
    o = np.argsort(-sc, kind='stable'); ww = w[:, o]; yy = y[o].astype(float)
    pos = ww @ yy; b = np.ceil(.1 * ww.sum(1)); before = np.cumsum(ww, axis=1) - ww
    sel = np.minimum(ww, np.maximum(b[:, None] - before, 0)); return (sel @ yy) / pos


def get(arm, y, s):
    if arm == 'UA7':
        p = (E22 / f'UA_s{s}/predictions.npz') if y == 2022 else (E2020 / str(y) / f'UA_s{s}/predictions.npz')
        return np.load(p, allow_pickle=False)['a_logit'].astype(float)
    return np.load(HERE / 'artifacts' / str(y) / f'{arm}_s{s}/fold3.npz', allow_pickle=False)['a_logit'].astype(float)


Y, SYM = {}, {}
for y in (2020, 2021):
    p = C.pack_for(y, range(4))['splits'][3]
    Y[y] = p['y'].astype(int); SYM[y] = p['keys'].Symbol.to_numpy()
W2 = K.shared_company_weights(SYM, 2000, 20261018)
curves2 = {(a, y): np.mean([curve(Y[y], get(a, y, s), W2[y]) for s in SEEDS], axis=0)
           for a in ('UA7', 'FiLM-L', 'CMA-G', 'CMA-L') for y in (2020, 2021)}
sys.path.insert(0, str(C.P / 'scripts'))
import train_audit_year2022 as fix22
fix22.install_redirected_loaders()
p = C.pack_for(2022, range(4))['splits'][3]
Y[2022] = p['y'].astype(int); SYM[2022] = p['keys'].Symbol.to_numpy()
W3 = K.shared_company_weights(SYM, 2000, 20261018)
curves3 = {(a, y): np.mean([curve(Y[y], get(a, y, s), W3[y]) for s in SEEDS], axis=0)
           for a in ('UA7', 'FiLM-L', 'CMA-G', 'CMA-L') for y in (2020, 2021, 2022)}

out = {}
for arm in ('FiLM-L', 'CMA-G', 'CMA-L'):
    d2 = np.mean([curves2[(arm, y)] - curves2[('UA7', y)] for y in (2020, 2021)], axis=0)
    d3 = np.mean([curves3[(arm, y)] - curves3[('UA7', y)] for y in (2020, 2021, 2022)], axis=0)
    out[arm] = dict(two_year=dict(point_pp=100 * float(np.mean(d2)),
                                  lower_pp=100 * float(np.quantile(d2, .025)),
                                  upper_pp=100 * float(np.quantile(d2, .975)),
                                  p_le_0=float(np.mean(d2 <= 0))),
                    three_year=dict(point_pp=100 * float(np.mean(d3)),
                                    lower_pp=100 * float(np.quantile(d3, .025)),
                                    upper_pp=100 * float(np.quantile(d3, .975)),
                                    p_le_0=float(np.mean(d3 <= 0))))
    print(arm, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out[arm].items()},
                          ensure_ascii=False))
(HERE / 'results' / 'final_ci.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
