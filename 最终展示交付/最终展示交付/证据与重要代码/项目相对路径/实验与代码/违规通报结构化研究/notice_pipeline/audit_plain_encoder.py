"""One predeclared parameter-matched ordinary X3 encoder, without branch search."""
from __future__ import annotations
import math
import torch
from torch import nn
from .audit_distributed import AuditDistributed

ENCODER_SEED_OFFSET = 3000


class PlainAuditEncoder(AuditDistributed):
    def __init__(self, formal, pack, budget, seed):
        super().__init__(formal, pack, budget, seed)
        input_dim = pack['splits'][0]['xs'][2].shape[1]
        output_dim = self.content_dims[2]
        branch_parameters = sum(p.numel() for p in self.base.enc[2].parameters()) + sum(p.numel() for p in self.audit.parameters())
        hidden = round((branch_parameters - output_dim) / (input_dim + output_dim + 1))
        assert hidden >= 1
        state = torch.random.get_rng_state()
        try:
            encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(.2),
                                    nn.Linear(hidden, output_dim), nn.ReLU())
            generator = torch.Generator(device='cpu').manual_seed(seed + ENCODER_SEED_OFFSET)
            for layer in encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
                    bound = 1 / math.sqrt(layer.in_features)
                    nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
        finally:
            torch.random.set_rng_state(state)
        self.base.enc[2] = encoder
        del self.audit
        self.plain_hidden = hidden
        self.branch_parameter_reference = branch_parameters
        self.plain_node_parameters = sum(p.numel() for p in encoder.parameters())
        self.node_parameter_difference = self.plain_node_parameters - branch_parameters
        self.encoder_seed = seed + ENCODER_SEED_OFFSET
        self.node3_initial_hash = formal.prev.digest(encoder.state_dict().items())
        self.initial_hash = formal.prev.digest(self.state_dict().items())
        self.initial_parameter_hash = formal.prev.digest(self.named_parameters())
        formal.prev.seed_all(seed)

    def forward(self, xs, industry, q, state):
        context = self.base.context(industry)
        hs = [self.base.enc[0](xs[0]), self.base.enc[1](xs[1]), self.base.enc[2](xs[2])]
        self.messages = [torch.cat([h, q[:, i:i + 1]], dim=1) for i, h in enumerate(hs)]
        self.base.messages = self.messages
        score = [self.base.score(torch.cat([projection(message[:, :d]), context,
                    self.base.party.weight[i].expand(len(message), -1)], dim=1))
                 for i, (message, projection, d) in enumerate(zip(self.messages, self.base.projections, self.content_dims))]
        alpha = torch.softmax(torch.cat(score, dim=1), dim=1)
        z = torch.cat([*[message * alpha[:, i:i + 1] for i, message in enumerate(self.messages)], context], dim=1)
        z = self.base.trunk(z)
        return self.base.any_head(z).squeeze(-1), self.base.type_head(z)
