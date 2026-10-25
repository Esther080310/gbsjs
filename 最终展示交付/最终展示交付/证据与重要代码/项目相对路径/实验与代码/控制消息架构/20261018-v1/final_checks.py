"""Per-year metric table + counterfactual for CMA-L. Each year is processed completely before
the 2022 loader redirect is installed, so earlier years always use the frozen cache."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import cma_lib as C  # noqa: E402

E22 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-audit-year2022-v1/distributed/D128'
E2020 = ROOT / '实验与代码/违规通报结构化研究/artifacts/experiments/20261016-evidence-v1/audit_information/distributed/D128'
_redirected = False


def ensure_2022():
    global _redirected
    if not _redirected:
        sys.path.insert(0, str(C.P / 'scripts'))
        import train_audit_year2022 as fix22
        fix22.install_redirected_loaders()
        _redirected = True


def pack4(year):
    if year == 2022:
        ensure_2022()
    return C.pack_for(year, range(4))


def score_of(arm, year, seed):
    if arm in ('CMA', 'CMA-L'):
        with np.load(HERE / 'artifacts' / str(year) / f'{arm}_s{seed}/fold3.npz', allow_pickle=False) as z:
            return z['a_logit'].astype(float)
    name = 'UA' if arm == 'UA7' else arm
    p = (E22 / f'{name}_s{seed}/predictions.npz') if year == 2022 else \
        (E2020 / str(year) / f'{name}_s{seed}/predictions.npz')
    with np.load(p, allow_pickle=False) as z:
        return z['a_logit'].astype(float)


rows, cf = [], {}
for year in (2020, 2021, 2022):
    split = pack4(year)['splits'][3]
    y = split['y'].astype(int)
    k = int(np.ceil(.1 * len(y)))
    for arm in ('U0', 'UA7', 'CMA', 'CMA-L'):
        rec, prec, ap, br, ll = [], [], [], [], []
        for s in C.SEEDS:
            sc = score_of(arm, year, s)
            order = np.argsort(-sc, kind='stable')
            hits = int(y[order[:k]].sum())
            prob = 1 / (1 + np.exp(-sc))
            rec.append(hits / y.sum()); prec.append(hits / k)
            ap.append(average_precision_score(y, sc))
            br.append(float(np.mean((prob - y) ** 2)))
            ll.append(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)))
        rows.append(dict(year=year, arm=arm, recall=float(np.mean(rec)), precision=float(np.mean(prec)),
                         ap=float(np.mean(ap)), brier=float(np.mean(br)), logloss=float(np.mean(ll))))
    # counterfactual for the FINAL model
    a = split['xs'][2][:, 16:23].copy()
    cats = np.argmax(a[:, 1:], axis=1) + 1
    x3 = split['xs'][2]
    probs = []
    for s in C.SEEDS:
        model = C.ControlMessageNet(C.shared.load_loader().formal, C.pack_for(year), s, 'linear')
        model.load_state_dict(torch.load(HERE / 'artifacts' / str(year) / f'CMA-L_s{s}/model.pt',
                                         map_location='cpu', weights_only=True)['state_dict'])
        model.eval()

        def run(x3v):
            with torch.no_grad():
                t = [torch.tensor(v) for v in split['xs']]
                t[2] = torch.tensor(x3v)
                logit, _ = model(t, torch.tensor(split['industry']), torch.tensor(split['q']),
                                 torch.tensor(split['state']))
                return torch.sigmoid(logit).numpy()
        cfa = a.copy(); cfa[:, 1:7] = 0.0; cfa[:, 1] = 1.0
        probs.append(run(x3) - run(np.concatenate([x3[:, :16], cfa], axis=1).astype(np.float32)))
    d = np.mean(probs, axis=0)
    nonstd = cats != 1
    cf[str(year)] = dict(mean_nonstandard=float(d[nonstd].mean()),
                         mean_standard=float(d[cats == 1].mean()),
                         max=float(d.max()), n_nonstandard=int(nonstd.sum()))
    print(json.dumps({'year': year, 'rows': int(len(y)), 'positives': int(y.sum()),
                      'cf': {k: round(v, 4) for k, v in cf[str(year)].items()}}, ensure_ascii=False), flush=True)

(HERE / 'results' / 'final_checks.json').write_text(
    json.dumps(dict(metrics=rows, counterfactual_cma_l=cf), ensure_ascii=False, indent=2), encoding='utf8')
for y in (2020, 2021, 2022):
    print(y, ' | '.join(f"{r['arm']}: R{100*r['recall']:.1f} P{100*r['precision']:.1f} "
                        f"AP{r['ap']:.3f} LL{r['logloss']:.3f}" for r in rows if r['year'] == y))
