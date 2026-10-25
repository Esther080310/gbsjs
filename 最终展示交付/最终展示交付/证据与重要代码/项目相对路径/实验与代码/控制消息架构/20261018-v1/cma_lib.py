"""Control-Message Architecture (CMA).

The audit node no longer sends a 21-float CONTENT message. It sends an 8-float CONTROL code
that modulates the finance node's representation (low-rank FiLM), and that same 8 floats are
the whole third-node message. Total message budget 128 -> 115 floats.

Rationale (empirical, from the E2 round): the audit increment is delivered through conditional
modulation of the finance representation, not through an independent additive message.
"""
from __future__ import annotations

import contextlib
import json
import sys
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
P = E / '违规通报结构化研究'
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P))
sys.path.insert(0, str(P / 'scripts'))
import train_auxiliary_semantic as shared  # noqa: E402
import train_audit_information as audit_line  # noqa: E402
from notice_pipeline.audit_distributed import FINANCE  # noqa: E402

AUDIT_DIM = 7
CONTROL_DIM = 8
CONTENT = (16, 88, CONTROL_DIM)          # finance keeps 88 (D128), node 3 shrinks to 8
BUDGET = sum(CONTENT) + 3                 # + one q float per node = 115
SEEDS = (42, 100, 17, 223, 483, 907)
YEARS = (2020, 2021)


def pack_for(year: int, folds=range(3)):
    pack = audit_line.make_pack(year, 'UA', folds=folds)
    width = pack['splits'][0]['xs'][2].shape[1] - 16
    assert width == AUDIT_DIM, f'审计块应为 7 列，实际 {width}'
    return pack


class ControlMessageNet(nn.Module):
    def __init__(self, formal, pack, seed, head: str = 'mlp'):
        assert head in ('mlp', 'linear')
        super().__init__()
        self.central, self.mode, self.messages = False, 'Plain', None
        self.content_dims = CONTENT
        self.budget = BUDGET
        self.payload_bytes = BUDGET * 4
        self.old_x3_dimension = pack['splits'][0]['xs'][2].shape[1] - AUDIT_DIM
        old = formal.RiskModel(pack, 32, 'Plain', seed)
        self.base = old.base
        prev = formal.prev
        prev.seed_all(seed)
        self.base.enc[0] = prev.mlp(pack['splits'][0]['xs'][0].shape[1], CONTENT[0], 64)
        self.base.enc[1] = prev.Groups(prev.grouping(pack['names'][1], 'S'), FINANCE[128], hidden=64)
        self.base.enc[2] = nn.Sequential(nn.Linear(AUDIT_DIM, CONTROL_DIM), nn.Tanh())
        self.base.projections = nn.ModuleList(nn.Linear(d, 16) for d in CONTENT)
        self.base.trunk = nn.Sequential(nn.Linear(BUDGET + 8, 64), nn.ReLU(), nn.Dropout(.2))
        self.base.content_dims = CONTENT
        state = torch.random.get_rng_state()
        if head == 'mlp':
            self.film = nn.Sequential(nn.Linear(CONTROL_DIM, 32), nn.ReLU(),
                                      nn.Linear(32, 2 * CONTENT[1]))
        else:                       # parameter-matched: single linear modulation
            self.film = nn.Linear(CONTROL_DIM, 2 * CONTENT[1])
        last = self.film[-1] if isinstance(self.film, nn.Sequential) else self.film
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        torch.random.set_rng_state(state)
        self.head = head
        self.initial_hash = prev.digest(self.state_dict().items())
        prev.seed_all(seed)

    def forward(self, xs, industry, q, state):
        context = self.base.context(industry)
        h1 = self.base.enc[0](xs[0])
        h2 = self.base.enc[1](xs[1])
        a = xs[2][:, self.old_x3_dimension:self.old_x3_dimension + AUDIT_DIM]
        r = a[:, :1]
        c = self.base.enc[2](a)                     # 8-float control code (the node-3 message)
        gamma, beta = self.film(c).chunk(2, dim=1)
        h2 = h2 + r * (gamma * h2 + beta)           # control modulates the finance representation
        self.messages = [torch.cat([h, q[:, i:i + 1]], dim=1) for i, h in enumerate((h1, h2, c))]
        self.base.messages = self.messages
        score = [self.base.score(torch.cat([proj(message[:, :d]), context,
                 self.base.party.weight[i].expand(len(message), -1)], dim=1))
                 for i, (message, proj, d) in enumerate(zip(self.messages, self.base.projections,
                                                            self.content_dims))]
        alpha = torch.softmax(torch.cat(score, dim=1), dim=1)
        z = torch.cat([*[message * alpha[:, i:i + 1] for i, message in enumerate(self.messages)],
                       context], dim=1)
        z = self.base.trunk(z)
        return self.base.any_head(z).squeeze(-1), self.base.type_head(z)

    def penalty(self):
        return torch.tensor(0.)


def fit(year: int, seed: int, arm: str = 'CMA') -> dict:
    formal = shared.load_loader().formal
    torch.set_num_threads(4)
    pack = pack_for(year)
    head = 'mlp' if arm == 'CMA' else 'linear'
    model = ControlMessageNet(formal, pack, seed, head)
    path = HERE / 'artifacts' / str(year) / f'{arm}_s{seed}'
    path.mkdir(parents=True, exist_ok=True)
    with (path / 'training.log').open('w') as log, contextlib.redirect_stdout(log):
        runtime = formal.training(model, pack, path, seed)
        raw = {}
        for f, split in enumerate(pack['splits']):
            raw[f'{f}_a_logit'], raw[f'{f}_b_logit'] = formal.logits(model, split)
        weights = formal.prev.cost_weights(pack['splits'][0])
        cal = shared.fit_calibration(formal, raw, pack, weights)
    pack4 = pack_for(year, range(4))
    a, b = formal.logits(model, pack4['splits'][3])
    a_logit = formal.exp.prob.apply_cal(a - np.log(weights[0]), cal['overall'])
    b_logit = formal.exp.prob.apply_cal(b - np.log(weights[1]), cal['types_shrink'])
    np.savez_compressed(path / 'fold3.npz', a_logit=a_logit, b_logit=b_logit)
    rec = {**runtime, 'year': year, 'arm': arm, 'seed': seed, 'head': head,
           'content_dims': list(CONTENT), 'budget_floats': BUDGET,
           'parameters': sum(v.numel() for v in model.parameters()),
           'initial_hash': shared.digest_state(model),
           'final_hash': shared.digest_state(model), 'x3_dim': int(pack['splits'][0]['xs'][2].shape[1])}
    (path / 'run.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf8')
    return rec
