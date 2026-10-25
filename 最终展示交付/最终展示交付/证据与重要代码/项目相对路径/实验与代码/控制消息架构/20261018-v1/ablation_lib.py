"""Two ablation arms for the main-model decision.

CMA-G : node 3 sends an 8-float governance summary PLUS the 8-float audit control code
        (16 floats, total budget 123). Isolates "did we lose accuracy by dropping governance
        content, or because 8 floats are too few?".

FiLM-L: keeps the original X3 content message (21 floats) and uses the audit ONLY to modulate
        the finance message with a single linear head. Budget stays 128, parameters +3%.
        This is the accuracy-neutral version of the structural idea.
"""
from __future__ import annotations

import contextlib
import json
import sys
import torch
from pathlib import Path
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cma_lib as C  # noqa: E402
from cma_lib import AUDIT_DIM, CONTROL_DIM, FINANCE, shared, audit_line, torch  # noqa: E402

GOV_SUMMARY = 8


class GovSummaryControl(nn.Module):
    """node3 = 8-float governance summary + 8-float control code; control modulates finance."""

    def __init__(self, formal, pack, seed):
        super().__init__()
        self.central, self.mode, self.messages = False, 'Plain', None
        self.content_dims = (16, 88, GOV_SUMMARY + CONTROL_DIM)
        self.budget = sum(self.content_dims) + 3
        self.payload_bytes = self.budget * 4
        self.old_x3_dimension = pack['splits'][0]['xs'][2].shape[1] - AUDIT_DIM
        old = formal.RiskModel(pack, 32, 'Plain', seed)
        self.base = old.base
        prev = formal.prev
        prev.seed_all(seed)
        self.base.enc[0] = prev.mlp(pack['splits'][0]['xs'][0].shape[1], 16, 64)
        self.base.enc[1] = prev.Groups(prev.grouping(pack['names'][1], 'S'), FINANCE[128], hidden=64)
        self.base.enc[2] = prev.Groups(pack['events'], (12, 9), hidden=16, event=True)
        self.gov = nn.Linear(21, GOV_SUMMARY)
        self.control = nn.Sequential(nn.Linear(AUDIT_DIM, CONTROL_DIM), nn.Tanh())
        self.base.projections = nn.ModuleList(nn.Linear(d, 16) for d in self.content_dims)
        self.base.trunk = nn.Sequential(nn.Linear(self.budget + 8, 64), nn.ReLU(), nn.Dropout(.2))
        self.base.content_dims = self.content_dims
        state = torch.random.get_rng_state()
        self.film = nn.Linear(CONTROL_DIM, 2 * 88)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        torch.random.set_rng_state(state)
        self.initial_hash = prev.digest(self.state_dict().items())
        prev.seed_all(seed)

    def _tail(self, h1, h2, h3, industry, q):
        context = self.base.context(industry)
        self.messages = [torch.cat([h, q[:, i:i + 1]], dim=1) for i, h in enumerate((h1, h2, h3))]
        self.base.messages = self.messages
        score = [self.base.score(torch.cat([proj(message[:, :d]), context,
                 self.base.party.weight[i].expand(len(message), -1)], dim=1))
                 for i, (message, proj, d) in enumerate(zip(self.messages, self.base.projections,
                                                            self.content_dims))]
        alpha = torch.softmax(torch.cat(score, dim=1), dim=1)
        z = torch.cat([*[m * alpha[:, i:i + 1] for i, m in enumerate(self.messages)], context], dim=1)
        z = self.base.trunk(z)
        return self.base.any_head(z).squeeze(-1), self.base.type_head(z)

    def forward(self, xs, industry, q, state):
        h1 = self.base.enc[0](xs[0])
        h2 = self.base.enc[1](xs[1])
        h_exec = self.base.enc[2](xs[2], state)
        a = xs[2][:, self.old_x3_dimension:self.old_x3_dimension + AUDIT_DIM]
        r = a[:, :1]
        c = self.control(a)
        gamma, beta = self.film(c).chunk(2, dim=1)
        h2 = h2 + r * (gamma * h2 + beta)
        g = torch.tanh(self.gov(h_exec))
        h3 = torch.cat([g, c], dim=1)
        return self._tail(h1, h2, h3, industry, q)

    def penalty(self):
        return torch.tensor(0.)


class FilmKeepX3(nn.Module):
    """Original X3 content message kept; audit only modulates finance (single linear head)."""

    def __init__(self, formal, pack, seed):
        super().__init__()
        self.central, self.mode, self.messages = False, 'Plain', None
        self.content_dims = (16, 88, 21)
        self.budget = sum(self.content_dims) + 3          # 128, same as UA7
        self.payload_bytes = self.budget * 4
        self.old_x3_dimension = pack['splits'][0]['xs'][2].shape[1] - AUDIT_DIM
        old = formal.RiskModel(pack, 32, 'Plain', seed)
        self.base = old.base
        prev = formal.prev
        prev.seed_all(seed)
        self.base.enc[0] = prev.mlp(pack['splits'][0]['xs'][0].shape[1], 16, 64)
        self.base.enc[1] = prev.Groups(prev.grouping(pack['names'][1], 'S'), FINANCE[128], hidden=64)
        self.base.enc[2] = prev.Groups(pack['events'], (12, 9), hidden=16, event=True)
        self.control = nn.Sequential(nn.Linear(AUDIT_DIM, CONTROL_DIM), nn.Tanh())
        self.base.projections = nn.ModuleList(nn.Linear(d, 16) for d in self.content_dims)
        self.base.trunk = nn.Sequential(nn.Linear(self.budget + 8, 64), nn.ReLU(), nn.Dropout(.2))
        self.base.content_dims = self.content_dims
        state = torch.random.get_rng_state()
        self.film = nn.Linear(CONTROL_DIM, 2 * 88)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        torch.random.set_rng_state(state)
        self.initial_hash = prev.digest(self.state_dict().items())
        prev.seed_all(seed)

    def forward(self, xs, industry, q, state):
        context = self.base.context(industry)
        h1 = self.base.enc[0](xs[0])
        h2 = self.base.enc[1](xs[1])
        h3 = self.base.enc[2](xs[2], state)
        a = xs[2][:, self.old_x3_dimension:self.old_x3_dimension + AUDIT_DIM]
        r = a[:, :1]
        c = self.control(a)
        gamma, beta = self.film(c).chunk(2, dim=1)
        h2 = h2 + r * (gamma * h2 + beta)
        self.messages = [torch.cat([h, q[:, i:i + 1]], dim=1) for i, h in enumerate((h1, h2, h3))]
        self.base.messages = self.messages
        score = [self.base.score(torch.cat([proj(message[:, :d]), context,
                 self.base.party.weight[i].expand(len(message), -1)], dim=1))
                 for i, (message, proj, d) in enumerate(zip(self.messages, self.base.projections,
                                                            self.content_dims))]
        alpha = torch.softmax(torch.cat(score, dim=1), dim=1)
        z = torch.cat([*[m * alpha[:, i:i + 1] for i, m in enumerate(self.messages)], context], dim=1)
        z = self.base.trunk(z)
        return self.base.any_head(z).squeeze(-1), self.base.type_head(z)

    def penalty(self):
        return torch.tensor(0.)


def build(arm, formal, pack, seed):
    return GovSummaryControl(formal, pack, seed) if arm == 'CMA-G' else FilmKeepX3(formal, pack, seed)


def fit(year: int, arm: str, seed: int) -> dict:
    formal = shared.load_loader().formal
    torch.set_num_threads(4)
    pack = C.pack_for(year)
    model = build(arm, formal, pack, seed)
    path = HERE / 'artifacts' / str(year) / f'{arm}_s{seed}'
    path.mkdir(parents=True, exist_ok=True)
    with (path / 'training.log').open('w') as log, contextlib.redirect_stdout(log):
        runtime = formal.training(model, pack, path, seed)
        raw = {}
        for f, split in enumerate(pack['splits']):
            raw[f'{f}_a_logit'], raw[f'{f}_b_logit'] = formal.logits(model, split)
        weights = formal.prev.cost_weights(pack['splits'][0])
        cal = shared.fit_calibration(formal, raw, pack, weights)
    pack4 = C.pack_for(year, range(4))
    a, b = formal.logits(model, pack4['splits'][3])
    a_logit = formal.exp.prob.apply_cal(a - np.log(weights[0]), cal['overall'])
    b_logit = formal.exp.prob.apply_cal(b - np.log(weights[1]), cal['types_shrink'])
    np.savez_compressed(path / 'fold3.npz', a_logit=a_logit, b_logit=b_logit)
    rec = {**runtime, 'year': year, 'arm': arm, 'seed': seed,
           'content_dims': list(model.content_dims), 'budget_floats': int(model.budget),
           'parameters': sum(v.numel() for v in model.parameters()),
           'final_hash': shared.digest_state(model)}
    (path / 'run.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf8')
    return rec


import numpy as np  # noqa: E402  (used in fit)
