"""FiLM-L logical wire protocol helpers.

This module does not evaluate labels or choose a communication policy.  It only
extracts the messages that would cross a star-topology deployment boundary and
provides deterministic FP32/FP16/INT8 round trips for the preflight audit.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTROL = ROOT / "实验与代码/控制消息架构/20261018-v1"
sys.path.insert(0, str(CONTROL))

import cma_lib as C  # noqa: E402
import ablation_lib as A  # noqa: E402

BLOCKS = ("market", "finance", "governance", "audit_control")
# Four logical blocks travel over three data-node uplinks.  In the current
# three-node project topology, governance and audit_control are both produced
# from X3 and can share one physical connection/round.
PHYSICAL_NODE = {
    "market": "x1",
    "finance": "x2",
    "governance": "x3",
    "audit_control": "x3",
}
PRECISION_CODE = {"fp32": 1, "fp16": 2, "int8": 3}
CODE_PRECISION = {v: k for k, v in PRECISION_CODE.items()}
HEADER = struct.Struct("<4sBBBBIHHI")  # 20 bytes, scales are pre-shared model metadata
MAGIC = b"FLMC"


def install_year(year: int) -> None:
    """Install the redirected 2022 loader before building that year's pack."""
    if year == 2022:
        sys.path.insert(0, str(C.P / "scripts"))
        import train_audit_year2022 as y22

        y22.install_redirected_loaders()


def pack_for(year: int, folds=range(4)):
    install_year(year)
    return C.pack_for(year, folds)


def load_model(year: int, seed: int, folds=range(4)):
    formal = C.shared.load_loader().formal
    pack = pack_for(year, folds)
    model = A.FilmKeepX3(formal, pack, seed)
    path = CONTROL / "artifacts" / str(year) / f"FiLM-L_s{seed}" / "model.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return formal, pack, model


@torch.no_grad()
def extract_messages(model, split: dict, rows: slice | np.ndarray) -> dict[str, np.ndarray]:
    """Extract the four logical messages before fusion-side FiLM.

    The finance node sends the unmodulated h2.  The audit source sends c and r;
    the fusion server applies h2' = h2 + r * (gamma(c) * h2 + beta(c)).
    """
    xs = [torch.as_tensor(x[rows], dtype=torch.float32) for x in split["xs"]]
    q = torch.as_tensor(split["q"][rows], dtype=torch.float32)
    state = torch.as_tensor(split["state"][rows], dtype=torch.float32)
    h1 = model.base.enc[0](xs[0])
    h2 = model.base.enc[1](xs[1])
    h3 = model.base.enc[2](xs[2], state)
    a = xs[2][:, model.old_x3_dimension:model.old_x3_dimension + C.AUDIT_DIM]
    r = a[:, :1]
    c = model.control(a)
    out = {
        "market": torch.cat([h1, q[:, 0:1]], dim=1),
        "finance": torch.cat([h2, q[:, 1:2]], dim=1),
        "governance": torch.cat([h3, q[:, 2:3]], dim=1),
        "audit_control": torch.cat([c, r], dim=1),
    }
    return {k: v.cpu().numpy().astype(np.float32, copy=False) for k, v in out.items()}


@torch.no_grad()
def fusion_from_messages(model, industry: np.ndarray, messages: dict[str, np.ndarray]):
    """Reproduce FiLM-L fusion from the four wire messages."""
    m1 = torch.as_tensor(messages["market"], dtype=torch.float32)
    m2 = torch.as_tensor(messages["finance"], dtype=torch.float32)
    m3 = torch.as_tensor(messages["governance"], dtype=torch.float32)
    ac = torch.as_tensor(messages["audit_control"], dtype=torch.float32)
    context = model.base.context(torch.as_tensor(industry, dtype=torch.long))
    h2, q2 = m2[:, :88], m2[:, 88:89]
    c, r = ac[:, :8], ac[:, 8:9]
    gamma, beta = model.film(c).chunk(2, dim=1)
    h2p = h2 + r * (gamma * h2 + beta)
    fused_messages = [m1, torch.cat([h2p, q2], dim=1), m3]
    score = [model.base.score(torch.cat([
        proj(message[:, :d]), context,
        model.base.party.weight[i].expand(len(message), -1)
    ], dim=1)) for i, (message, proj, d) in enumerate(zip(
        fused_messages, model.base.projections, model.content_dims
    ))]
    alpha = torch.softmax(torch.cat(score, dim=1), dim=1)
    z = torch.cat([
        *[message * alpha[:, i:i + 1] for i, message in enumerate(fused_messages)],
        context,
    ], dim=1)
    z = model.base.trunk(z)
    return (model.base.any_head(z).squeeze(-1).cpu().numpy(),
            model.base.type_head(z).cpu().numpy())


def fit_scales(messages: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-channel symmetric INT8 scales; fitted only on non-evaluation messages."""
    scales = {}
    for block, x in messages.items():
        values = x[:, :8] if block == "audit_control" else x
        m = np.max(np.abs(values.astype(np.float64)), axis=0)
        scales[block] = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    return scales


def quantize_dequantize(block: str, x: np.ndarray, precision: str,
                        scale: np.ndarray | None = None) -> np.ndarray:
    """Deterministic round trip.  Audit match r is always transmitted exactly."""
    assert precision in PRECISION_CODE
    r = x[:, 8:9].copy() if block == "audit_control" else None
    values = x[:, :8] if block == "audit_control" else x
    if precision == "fp32":
        decoded = values.astype("<f4", copy=False).copy()
    elif precision == "fp16":
        decoded = values.astype("<f2").astype(np.float32)
    else:
        assert scale is not None and scale.shape == (values.shape[1],)
        code = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
        decoded = code.astype(np.float32) * scale
    return np.concatenate([decoded, r], axis=1) if r is not None else decoded


def serialize_block(block: str, x: np.ndarray, precision: str,
                    scale: np.ndarray | None = None) -> bytes:
    """Serialize one cohort frame; fixed scales are referenced, not repeated."""
    block_id = BLOCKS.index(block)
    r = x[:, 8].astype(np.uint8, copy=False) if block == "audit_control" else None
    values = x[:, :8] if block == "audit_control" else x
    if precision == "fp32":
        body = values.astype("<f4", copy=False).tobytes(order="C")
    elif precision == "fp16":
        body = values.astype("<f2").tobytes(order="C")
    else:
        assert scale is not None and scale.shape == (values.shape[1],)
        body = np.clip(np.rint(values / scale), -127, 127).astype(np.int8).tobytes(order="C")
    flags = 1 if r is not None else 0
    if r is not None:
        body += r.tobytes(order="C")
    head = HEADER.pack(MAGIC, 1, block_id, PRECISION_CODE[precision], flags,
                       len(x), x.shape[1], 0, len(body))
    return head + body


def deserialize_block(frame: bytes, scale: np.ndarray | None = None):
    magic, version, block_id, code, flags, n, dim, _reserved, body_len = HEADER.unpack(
        frame[:HEADER.size]
    )
    assert magic == MAGIC and version == 1 and len(frame) == HEADER.size + body_len
    block, precision = BLOCKS[block_id], CODE_PRECISION[code]
    has_r = bool(flags & 1)
    value_dim = dim - 1 if has_r else dim
    item = {"fp32": 4, "fp16": 2, "int8": 1}[precision]
    value_bytes = n * value_dim * item
    body = memoryview(frame)[HEADER.size:]
    dtype = {"fp32": "<f4", "fp16": "<f2", "int8": np.int8}[precision]
    raw = np.frombuffer(body[:value_bytes], dtype=dtype).reshape(n, value_dim)
    if precision == "int8":
        assert scale is not None and scale.shape == (value_dim,)
        values = raw.astype(np.float32) * scale
    else:
        values = raw.astype(np.float32)
    if has_r:
        r = np.frombuffer(body[value_bytes:], dtype=np.uint8).reshape(n, 1).astype(np.float32)
        values = np.concatenate([values, r], axis=1)
    return block, precision, values

