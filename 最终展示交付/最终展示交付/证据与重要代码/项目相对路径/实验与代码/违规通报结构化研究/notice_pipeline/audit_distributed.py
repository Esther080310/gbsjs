"""Matched audit information at 32/64/128 total float messages.

The old executive event state gates only the old executive branch. The separate
source audit branch is gated by audit match r and never by executive state.
"""
from __future__ import annotations
import math
import torch
from torch import nn

CONTENT = {32: (7, 15, 7), 64: (12, 36, 13), 128: (16, 88, 21)}
FINANCE = {32: (5, 5, 5), 64: (12, 12, 12), 128: (30, 29, 29)}
EXECUTIVE = {32: (4, 3), 64: (7, 6), 128: (12, 9)}
AUDIT_SEED_OFFSET = 2000


def make_audit_branch(seed, output_dim):
    state = torch.random.get_rng_state()
    try:
        branch = nn.Sequential(nn.Linear(7, 16), nn.ReLU(), nn.Linear(16, output_dim), nn.Tanh())
        generator = torch.Generator(device='cpu').manual_seed(seed + AUDIT_SEED_OFFSET)
        for layer in branch:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
                bound = 1 / math.sqrt(layer.in_features)
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
    finally:
        torch.random.set_rng_state(state)
    return branch


class AuditDistributed(nn.Module):
    def __init__(self, formal, pack, budget, seed):
        super().__init__()
        if budget not in CONTENT:
            raise ValueError('Total message budget must be 32, 64 or 128')
        self.central, self.mode, self.messages = False, 'Plain', None
        self.budget = budget
        self.content_dims = CONTENT[budget]
        self.payload_bytes = budget * 4
        self.old_x3_dimension = pack['splits'][0]['xs'][2].shape[1] - 7
        assert self.old_x3_dimension == 16
        # In D32 none of the original same-shape components are rebuilt. The
        # old event indices select the original sixteen X3 input columns only.
        old = formal.RiskModel(pack, 32, 'Plain', seed)
        self.base = old.base
        prev = formal.prev
        if budget != 32:
            prev.seed_all(seed)
            self.base.enc[0] = prev.mlp(pack['splits'][0]['xs'][0].shape[1], self.content_dims[0], 64)
            self.base.enc[1] = prev.Groups(prev.grouping(pack['names'][1], 'S'), FINANCE[budget], hidden=64)
            self.base.enc[2] = prev.Groups(pack['events'], EXECUTIVE[budget], hidden=16, event=True)
            self.base.projections = nn.ModuleList(nn.Linear(d, 16) for d in self.content_dims)
            self.base.trunk = nn.Sequential(nn.Linear(budget + 8, 64), nn.ReLU(), nn.Dropout(.2))
            self.base.content_dims = self.content_dims
        self.old_components_initial_hash = prev.digest(self.base.state_dict().items())
        before = torch.random.get_rng_state().clone()
        self.audit = make_audit_branch(seed, self.content_dims[2])
        assert torch.equal(before, torch.random.get_rng_state())
        self.audit_initial_hash = prev.digest(self.audit.state_dict().items())
        self.initial_hash = prev.digest(self.state_dict().items())
        self.initial_parameter_hash = prev.digest(self.named_parameters())
        prev.seed_all(seed)

    def original_s32_state_dict(self):
        """Original formal.RiskModel keys, excluding the newly registered branch."""
        return {'base.' + key: value for key, value in self.base.state_dict().items()}

    def forward(self, xs, industry, q, state):
        context = self.base.context(industry)
        # Preserve the original h1/h2/h3 invocation order and dropout stream.
        h1 = self.base.enc[0](xs[0])
        h2 = self.base.enc[1](xs[1])
        h_exec = self.base.enc[2](xs[2], state)
        audit_input = xs[2][:, self.old_x3_dimension:self.old_x3_dimension + 7]
        r = audit_input[:, :1]
        h3 = h_exec + r * self.audit(audit_input) if r.any() else h_exec
        self.messages = [torch.cat([h, q[:, i:i + 1]], dim=1) for i, h in enumerate((h1, h2, h3))]
        self.base.messages = self.messages
        score = [self.base.score(torch.cat([projection(message[:, :d]), context,
                    self.base.party.weight[i].expand(len(message), -1)], dim=1))
                 for i, (message, projection, d) in enumerate(zip(self.messages, self.base.projections, self.content_dims))]
        alpha = torch.softmax(torch.cat(score, dim=1), dim=1)
        z = torch.cat([*[message * alpha[:, i:i + 1] for i, message in enumerate(self.messages)], context], dim=1)
        z = self.base.trunk(z)
        return self.base.any_head(z).squeeze(-1), self.base.type_head(z)

    def penalty(self):
        return torch.tensor(0.)
