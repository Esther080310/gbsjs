"""Counterfactual audit swap: what does the audit opinion do to each firm's score?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cma_lib as C  # noqa: E402

formal = C.shared.load_loader().formal
torch.set_num_threads(4)
rows = []
for year in C.YEARS:
    pack4 = C.pack_for(year, range(4))
    split = pack4['splits'][3]
    a = split['xs'][2][:, 16:23].copy()
    cats = np.argmax(a[:, 1:], axis=1) + 1
    base_x3 = split['xs'][2]
    probs = {s: [] for s in C.SEEDS}
    for s in C.SEEDS:
        model = C.ControlMessageNet(formal, C.pack_for(year), s)
        model.load_state_dict(torch.load(HERE / 'artifacts' / str(year) / f'CMA_s{s}/model.pt',
                                         map_location='cpu', weights_only=True)['state_dict'])
        model.eval()

        def score(x3):
            with torch.no_grad():
                t = [torch.tensor(v) for v in split['xs']]
                t[2] = torch.tensor(x3)
                logit, _ = model(t, torch.tensor(split['industry']),
                                 torch.tensor(split['q']), torch.tensor(split['state']))
                return torch.sigmoid(logit).numpy()

        cf = a.copy()
        cf[:, 1:7] = 0.0
        cf[:, 1] = 1.0                      # force "standard unqualified"
        x3_cf = np.concatenate([base_x3[:, :16], cf], axis=1).astype(np.float32)
        probs[s] = [score(base_x3), score(x3_cf)]
    p_act = np.mean([p[0] for p in probs.values()], axis=0)
    p_std = np.mean([p[1] for p in probs.values()], axis=0)
    d = p_act - p_std
    order = np.argsort(-d)
    nonstd = np.flatnonzero(cats != 1)
    rows.append(dict(year=year, n=int(len(d)),
                     mean_delta_nonstandard=float(d[nonstd].mean()) if len(nonstd) else 0.0,
                     mean_delta_standard=float(d[cats == 1].mean()),
                     max_delta=float(d.max()),
                     top=[dict(symbol=str(split['keys'].Symbol.to_numpy()[i]),
                               category=int(cats[i]), delta=float(d[i])) for i in order[:8]]))
    print(json.dumps(rows[-1], ensure_ascii=False))
(HERE / 'results' / 'counterfactual.json').write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
