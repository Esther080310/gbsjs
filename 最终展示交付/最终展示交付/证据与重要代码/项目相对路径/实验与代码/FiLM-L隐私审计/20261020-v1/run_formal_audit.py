"""Frozen FiLM-L representation inversion audit.

The attack target is the model's preprocessed node input, never the raw CSMAR table.
Fold 0 trains attackers, fold 1 selects/early-stops, fold 2 is not loaded, and fold 3 is
opened only for the final evaluation.  Results are checkpointed per year/scope/view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import linalg, stats
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.exceptions import ConvergenceWarning
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTROL = ROOT / "实验与代码/控制消息架构/20261018-v1"
NOTICE = ROOT / "实验与代码/违规通报结构化研究"
sys.path[:0] = [str(CONTROL), str(NOTICE), str(NOTICE / "scripts")]

import cma_lib as C  # noqa: E402
import ablation_lib as A  # noqa: E402
from audit_artifacts import (  # noqa: E402
    build_manifest,
    build_summaries,
    write_json_atomic,
    write_summaries,
)

YEARS = (2020, 2021, 2022)
SEEDS = tuple(C.SEEDS)
ATTACK_SEED = 20261020
ALPHAS = (0.1, 1.0, 10.0, 100.0)
BOOT_RECON = 500
BOOT_AUC = 200

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
warnings.filterwarnings("ignore", message="Mean of empty slice")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def extract(model: A.FilmKeepX3, split: dict, batch_size: int = 4096) -> dict[str, np.ndarray]:
    """Extract the exact pre- and post-FiLM messages without changing model parameters."""
    model.eval()
    output: dict[str, list[np.ndarray]] = {k: [] for k in
        ("h1", "h2", "h2_post", "h3", "c", "r", "q", "industry")}
    n = len(split["y"])
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xs = [torch.as_tensor(x[start:stop]) for x in split["xs"]]
        q = torch.as_tensor(split["q"][start:stop])
        state = torch.as_tensor(split["state"][start:stop])
        industry = torch.as_tensor(split["industry"][start:stop])
        h1 = model.base.enc[0](xs[0])
        h2 = model.base.enc[1](xs[1])
        h3 = model.base.enc[2](xs[2], state)
        audit = xs[2][:, model.old_x3_dimension:model.old_x3_dimension + C.AUDIT_DIM]
        r = audit[:, :1]
        c = model.control(audit)
        gamma, beta = model.film(c).chunk(2, dim=1)
        h2_post = h2 + r * (gamma * h2 + beta)
        # The first batch proves the manual transcript equals the model's own messages.
        if start == 0:
            _ = model(xs, industry, q, state)
            expected = (torch.cat([h1, q[:, :1]], 1),
                        torch.cat([h2_post, q[:, 1:2]], 1),
                        torch.cat([h3, q[:, 2:3]], 1))
            for left, right in zip(expected, model.messages):
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        for key, value in (("h1", h1), ("h2", h2), ("h2_post", h2_post),
                           ("h3", h3), ("c", c), ("r", r), ("q", q),
                           ("industry", industry[:, None].float())):
            output[key].append(value.detach().cpu().numpy().astype("f4", copy=False))
    return {key: np.concatenate(parts, axis=0) for key, parts in output.items()}


def build_views(seed_views: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    one = seed_views[42]
    single = {
        "h1": one["h1"], "h2": one["h2"],
        "h2_q": np.concatenate([one["h2"], one["q"][:, 1:2]], 1),
        "h2_cr": np.concatenate([one["h2"], one["c"], one["r"]], 1),
        "h2_qcr": np.concatenate([one["h2"], one["q"][:, 1:2], one["c"], one["r"]], 1),
        "h2_post": one["h2_post"], "h3": one["h3"],
        "full_deployment": np.concatenate(
            [one[k] for k in ("h1", "h2", "h3", "q", "c", "r", "industry")], 1),
        "full_implementation": np.concatenate(
            [one[k] for k in ("h1", "h2_post", "h3", "q", "industry")], 1),
    }
    joint = {
        "h1": np.concatenate([seed_views[s]["h1"] for s in SEEDS], 1),
        "h2": np.concatenate([seed_views[s]["h2"] for s in SEEDS], 1),
        "h2_q": np.concatenate(
            [*[seed_views[s]["h2"] for s in SEEDS], one["q"][:, 1:2]], 1),
        "h2_cr": np.concatenate(
            [*[seed_views[s]["h2"] for s in SEEDS],
             *[seed_views[s]["c"] for s in SEEDS], one["r"]], 1),
        "h2_qcr": np.concatenate(
            [*[seed_views[s]["h2"] for s in SEEDS], one["q"][:, 1:2],
             *[seed_views[s]["c"] for s in SEEDS], one["r"]], 1),
        "h2_post": np.concatenate([seed_views[s]["h2_post"] for s in SEEDS], 1),
        "h3": np.concatenate([seed_views[s]["h3"] for s in SEEDS], 1),
        "full_deployment": np.concatenate(
            [*[np.concatenate([seed_views[s][k] for k in ("h1", "h2", "h3", "c")], 1)
               for s in SEEDS], one["q"], one["r"], one["industry"]], 1),
        "full_implementation": np.concatenate(
            [*[np.concatenate([seed_views[s][k] for k in ("h1", "h2_post", "h3")], 1)
               for s in SEEDS], one["q"], one["industry"]], 1),
    }
    expected_single = {"h1": 16, "h2": 88, "h2_q": 89, "h2_cr": 97,
                       "h2_qcr": 98, "h2_post": 88, "h3": 21,
                       "full_deployment": 138, "full_implementation": 129}
    expected_joint = {"h1": 96, "h2": 528, "h2_q": 529, "h2_cr": 577,
                      "h2_qcr": 578, "h2_post": 528, "h3": 126,
                      "full_deployment": 803, "full_implementation": 754}
    for key, width in expected_single.items():
        assert single[key].shape[1] == width, (key, single[key].shape)
    for key, width in expected_joint.items():
        assert joint[key].shape[1] == width, (key, joint[key].shape)
    return {"single42": single, "joint6": joint}


def build_target(split: dict, names: list[list[str]], view: str) -> tuple[np.ndarray, list[str], list[tuple[int, int, str]]]:
    x1, x2, x3 = split["xs"][0], split["xs"][1], split["xs"][2][:, :16]
    if view == "h1":
        return x1.astype("f4"), list(names[0]), [(0, x1.shape[1], "x1")]
    if view in ("h2", "h2_q", "h2_cr", "h2_qcr", "h2_post"):
        return x2.astype("f4"), list(names[1]), [(0, x2.shape[1], "x2")]
    if view == "h3":
        return x3.astype("f4"), list(names[2][:16]), [(0, x3.shape[1], "x3")]
    joined = np.concatenate([x1, x2, x3], 1).astype("f4")
    out_names = ([f"x1::{n}" for n in names[0]] + [f"x2::{n}" for n in names[1]] +
                 [f"x3::{n}" for n in names[2][:16]])
    a, b = x1.shape[1], x1.shape[1] + x2.shape[1]
    return joined, out_names, [(0, a, "x1"), (a, b, "x2"), (b, joined.shape[1], "x3")]


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(0, dtype=np.float64)
        scale = x.std(0, dtype=np.float64)
        scale[scale < 1e-7] = 1.0
        return cls(mean.astype("f4"), scale.astype("f4"))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype("f4")

    def undo(self, z: np.ndarray) -> np.ndarray:
        return (z * self.scale + self.mean).astype("f4")


def macro_nrmse(y: np.ndarray, pred: np.ndarray, train_scale: np.ndarray) -> float:
    per = np.sqrt(np.mean((y.astype("f8") - pred.astype("f8")) ** 2, axis=0)) / train_scale
    return float(np.mean(per))


def detailed_metrics(y: np.ndarray, pred: np.ndarray, train_scale: np.ndarray,
                     names: list[str]) -> tuple[dict, pd.DataFrame]:
    error = y.astype("f8") - pred.astype("f8")
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    nrmse = rmse / train_scale
    denom = np.sum((y.astype("f8") - y.mean(0, dtype=np.float64)) ** 2, axis=0)
    r2 = np.full(y.shape[1], np.nan)
    valid = denom > 1e-12
    r2[valid] = 1.0 - np.sum(error[:, valid] ** 2, axis=0) / denom[valid]
    spear = []
    for j in range(y.shape[1]):
        if np.std(y[:, j]) < 1e-8 or np.std(pred[:, j]) < 1e-8:
            spear.append(np.nan)
        else:
            spear.append(float(stats.spearmanr(y[:, j], pred[:, j]).statistic))
    binary = np.array([("missing::" in n) or ("=" in n) for n in names])
    result = {
        "macro_nrmse_all": float(np.mean(nrmse)),
        "macro_nrmse_numeric": float(np.mean(nrmse[~binary])) if (~binary).any() else None,
        "macro_nrmse_binary": float(np.mean(nrmse[binary])) if binary.any() else None,
        "macro_r2": float(np.nanmean(r2)),
        "macro_spearman": float(np.nanmean(spear)),
        "fields": int(len(names)), "binary_fields": int(binary.sum()),
    }
    fields = pd.DataFrame({"field": names, "rmse": rmse, "nrmse": nrmse,
                           "r2": r2, "spearman": spear, "binary_named": binary})
    return result, fields


class RidgeAttack:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeAttack":
        self.xs, self.ys = Standardizer.fit(x), Standardizer.fit(y)
        a, b = self.xs.apply(x).astype("f8"), self.ys.apply(y).astype("f8")
        gram = a.T @ a
        gram.flat[::gram.shape[0] + 1] += self.alpha
        self.coef = linalg.solve(gram, a.T @ b, assume_a="pos", check_finite=False).astype("f4")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ys.undo(self.xs.apply(x) @ self.coef)


def fit_ridge(x0, y0, x1, y1, smoke=False):
    candidates = ALPHAS[:2] if smoke else ALPHAS
    scale = Standardizer.fit(y0).scale
    rows = []
    best = None
    for alpha in candidates:
        model = RidgeAttack(alpha).fit(x0, y0)
        score = macro_nrmse(y1, model.predict(x1), scale)
        rows.append({"alpha": alpha, "selection_nrmse": score})
        if best is None or score < best[0]:
            best = (score, model)
    return best[1], rows


class AttackMLP(nn.Module):
    def __init__(self, p: int, q: int):
        super().__init__()
        width = min(256, 2 * p)
        self.net = nn.Sequential(nn.Linear(p, width), nn.ReLU(), nn.Linear(width, q))

    def forward(self, x):
        return self.net(x)


def predict_mlp(model: AttackMLP, xs: Standardizer, ys: Standardizer, x: np.ndarray,
                batch_size=4096) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        z = xs.apply(x)
        for start in range(0, len(z), batch_size):
            pieces.append(model(torch.from_numpy(z[start:start + batch_size])).numpy())
    return ys.undo(np.concatenate(pieces, 0))


def fit_mlp(x0, y0, x1, y1, smoke=False):
    seed_all(ATTACK_SEED)
    xs, ys = Standardizer.fit(x0), Standardizer.fit(y0)
    model = AttackMLP(x0.shape[1], y0.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader_gen = torch.Generator().manual_seed(ATTACK_SEED)
    loader = DataLoader(TensorDataset(torch.from_numpy(xs.apply(x0)), torch.from_numpy(ys.apply(y0))),
                        batch_size=1024, shuffle=True, generator=loader_gen)
    epochs, patience = (2, 1) if smoke else (30, 4)
    scale = ys.scale
    best_score, best_epoch, best_state, stale = math.inf, 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        pred = predict_mlp(model, xs, ys, x1)
        score = macro_nrmse(y1, pred, scale)
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)),
                        "selection_nrmse": score})
        if score < best_score - 1e-5:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, xs, ys, {"best_epoch": best_epoch, "selection_nrmse": best_score,
                           "history": history}


def pca_reference(y0, y3, dimension, ys: Standardizer):
    d = min(int(dimension), y0.shape[1], len(y0) - 1)
    if d == y0.shape[1]:
        return y3.copy(), {"components": d, "identity_limit": True}
    pca = PCA(n_components=d, svd_solver="randomized", random_state=ATTACK_SEED)
    pca.fit(ys.apply(y0))
    pred = ys.undo(pca.inverse_transform(pca.transform(ys.apply(y3))).astype("f4"))
    return pred, {"components": d, "identity_limit": False,
                  "explained_variance": float(pca.explained_variance_ratio_.sum())}


def random_projection_reference(y0, y1, y3, dimension, ys: Standardizer, smoke=False):
    d = min(int(dimension), y0.shape[1])
    projector = GaussianRandomProjection(n_components=d, random_state=ATTACK_SEED)
    z0 = projector.fit_transform(ys.apply(y0)).astype("f4")
    z1 = projector.transform(ys.apply(y1)).astype("f4")
    z3 = projector.transform(ys.apply(y3)).astype("f4")
    decoder, selection = fit_ridge(z0, y0, z1, y1, smoke=smoke)
    return decoder.predict(z3), {"components": d, "ridge_selection": selection}


def attribute_specs(names: list[str], y0: np.ndarray, meta: list[dict], segments):
    specs = []
    for j, name in enumerate(names):
        base = name.split("::", 1)[-1] if name.startswith(("x1::", "x2::", "x3::")) else name
        if base.startswith("missing::") or "=" in base:
            values = np.unique(y0[:, j])
            if len(values) == 2:
                specs.append({"field": name, "index": j, "kind": "encoded_binary",
                              "threshold": float(values.mean()), "scale": 1.0, "mean": 0.0})
    # First five X3 fields are event counts; inverse their frozen scaler and test raw count > 0.
    for start, stop, node in segments:
        if node != "x3":
            continue
        for local in range(5):
            j = start + local
            specs.append({"field": names[j] + "::event_present", "index": j,
                          "kind": "event_present", "threshold": 0.5,
                          "scale": float(meta[2]["scaler_scale"][local]),
                          "mean": float(meta[2]["scaler_mean"][local])})
    return specs


def attribute_metrics(y, pred, specs):
    rows = []
    for spec in specs:
        j = spec["index"]
        truth_value = y[:, j] * spec["scale"] + spec["mean"]
        score = pred[:, j] * spec["scale"] + spec["mean"]
        truth = truth_value > spec["threshold"]
        if truth.min() == truth.max():
            continue
        auc = roc_auc_score(truth, score)
        rows.append({"field": spec["field"], "kind": spec["kind"], "auc": float(auc),
                     "positive": int(truth.sum()), "n": int(len(truth))})
    return rows


def attribute_truth(y: np.ndarray, specs: list[dict]) -> np.ndarray:
    if not specs:
        return np.empty((len(y), 0), dtype=np.int8)
    cols = []
    for spec in specs:
        j = spec["index"]
        value = y[:, j] * spec["scale"] + spec["mean"]
        cols.append(value > spec["threshold"])
    return np.stack(cols, axis=1).astype(np.int8)


def fit_direct_attribute_attack(x0, y0, x1, y1, x3, y3, specs, smoke=False):
    """Direct multi-output L2-logistic property attack.

    A single linear torch layer estimates all registered properties jointly.  This
    is mathematically a collection of logistic regressions, evaluated in one
    matrix multiplication so X2's many missingness flags remain feasible.
    """
    truth0 = attribute_truth(y0, specs)
    truth1 = attribute_truth(y1, specs)
    truth3 = attribute_truth(y3, specs)
    variable = np.array([(truth0[:, j].min() != truth0[:, j].max())
                         for j in range(truth0.shape[1])], dtype=bool)
    if not variable.any():
        return [], np.empty((len(y3), 0), dtype="f4"), truth3[:, :0], {
            "solver": "multi-output linear logistic head, AdamW L2", "variable_fields": 0}
    xs = Standardizer.fit(x0)
    seed_all(ATTACK_SEED + 91)
    attack = nn.Linear(x0.shape[1], int(variable.sum()))
    optimizer = torch.optim.AdamW(attack.parameters(), lr=1e-2, weight_decay=1e-4)
    t0 = truth0[:, variable].astype("f4")
    t1 = truth1[:, variable].astype("f4")
    positive = np.maximum(t0.sum(0), 1.0)
    pos_weight = np.clip((len(t0) - positive) / positive, 1.0, 20.0).astype("f4")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight))
    generator = torch.Generator().manual_seed(ATTACK_SEED + 91)
    loader = DataLoader(TensorDataset(torch.from_numpy(xs.apply(x0)), torch.from_numpy(t0)),
                        batch_size=2048, shuffle=True, generator=generator)
    xv = torch.from_numpy(xs.apply(x1)); yv = torch.from_numpy(t1)
    epochs, patience = ((2, 1) if smoke else (20, 4))
    best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        attack.train(); losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(attack(xb), yb)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        attack.eval()
        with torch.no_grad():
            val_loss = float(criterion(attack(xv), yv))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "selection_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in attack.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    attack.load_state_dict(best_state); attack.eval()
    with torch.no_grad():
        score = torch.sigmoid(attack(torch.from_numpy(xs.apply(x3)))).numpy().astype("f4")
    used_specs = [spec for spec, keep in zip(specs, variable) if keep]
    used_truth = truth3[:, variable]
    rows = []
    for j, spec in enumerate(used_specs):
        truth = used_truth[:, j]
        if truth.min() == truth.max():
            continue
        rows.append({"field": spec["field"], "kind": spec["kind"],
                     "auc": float(roc_auc_score(truth, score[:, j])),
                     "positive": int(truth.sum()), "n": int(len(truth)),
                     "score_column": j})
    return rows, score, used_truth, {
        "solver": "multi-output linear logistic head, AdamW L2",
        "learning_rate": 1e-2, "weight_decay": 1e-4,
        "positive_weight_cap": 20.0, "best_epoch": best_epoch,
        "selection_loss": best_loss, "history": history,
        "registered_fields": int(len(specs)), "variable_train_fields": int(variable.sum()),
        "evaluable_test_fields": int(len(rows)),
    }


def reconstruction_bootstrap(y, pred, constant, scale, company_index, weights):
    attack_sq = (y.astype("f8") - pred.astype("f8")) ** 2
    constant_sq = (y.astype("f8") - constant.astype("f8")) ** 2
    rows = weights[:, company_index].astype("f8")
    out = []
    for start in range(0, len(rows), 25):
        w = rows[start:start + 25]
        denominator = np.maximum(w.sum(1), 1.0)[:, None]
        a = np.mean(np.sqrt((w @ attack_sq) / denominator) / scale[None, :], axis=1)
        c = np.mean(np.sqrt((w @ constant_sq) / denominator) / scale[None, :], axis=1)
        out.extend((c - a).tolist())
    return np.asarray(out, dtype="f4")


def auc_bootstrap(y, pred, specs, company_index, weights, repeats=BOOT_AUC):
    reps = []
    for weight in weights[:repeats]:
        row_weight = weight[company_index]
        field_aucs = []
        for spec in specs:
            j = spec["index"]
            truth = (y[:, j] * spec["scale"] + spec["mean"]) > spec["threshold"]
            if truth.min() == truth.max() or row_weight[truth].sum() == 0 or row_weight[~truth].sum() == 0:
                continue
            score = pred[:, j] * spec["scale"] + spec["mean"]
            field_aucs.append(roc_auc_score(truth, score, sample_weight=row_weight))
        reps.append(float(np.mean(field_aucs)) if field_aucs else np.nan)
    return np.asarray(reps, dtype="f4")


def direct_auc_bootstrap(truth, scores, rows, company_index, weights, repeats):
    reps = []
    valid_columns = [(int(row["score_column"]), row) for row in rows]
    for weight in weights[:repeats]:
        row_weight = weight[company_index]
        values = []
        for j, _row in valid_columns:
            label = truth[:, j]
            if label.min() == label.max() or row_weight[label == 1].sum() == 0 or \
                    row_weight[label == 0].sum() == 0:
                continue
            values.append(roc_auc_score(label, scores[:, j], sample_weight=row_weight))
        reps.append(float(np.mean(values)) if values else np.nan)
    return np.asarray(reps, dtype="f4")


def interval(values, null=0.0):
    finite = np.asarray(values)[np.isfinite(values)]
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {"estimate": float(np.mean(finite)), "ci95": [float(lo), float(hi)],
            "null": null, "lower_above_null": bool(lo > null), "replicates": int(len(finite))}


def attack_one(year, scope, view, folds, views, names, meta, global_company_to_index,
               boot_weights, output_dir, smoke=False, primary=False):
    x0, x1, x3 = (views[f][scope][view] for f in (0, 1, 3))
    y0, target_names, segments = build_target(folds[0], names, view)
    y1, _, _ = build_target(folds[1], names, view)
    y3, _, _ = build_target(folds[3], names, view)
    assert len(x0) == len(y0) and len(x1) == len(y1) and len(x3) == len(y3)
    assert np.isfinite(x0).all() and np.isfinite(y0).all()
    started = time.perf_counter()
    ys = Standardizer.fit(y0)
    scale = ys.scale.astype("f8")
    constant3 = np.broadcast_to(ys.mean, y3.shape).copy()

    ridge, ridge_selection = fit_ridge(x0, y0, x1, y1, smoke=smoke)
    ridge1, ridge3 = ridge.predict(x1), ridge.predict(x3)
    mlp, mx, my, mlp_record = fit_mlp(x0, y0, x1, y1, smoke=smoke)
    mlp1, mlp3 = predict_mlp(mlp, mx, my, x1), predict_mlp(mlp, mx, my, x3)
    selection = {"ridge": macro_nrmse(y1, ridge1, scale),
                 "mlp": macro_nrmse(y1, mlp1, scale)}
    chosen = min(selection, key=selection.get)
    chosen3 = ridge3 if chosen == "ridge" else mlp3

    predictions = {"constant": constant3, "ridge": ridge3, "mlp": mlp3}
    pca_record = rp_record = {"not_run_in_primary_stage": True}
    if not primary:
        pca3, pca_record = pca_reference(y0, y3, x0.shape[1], ys)
        rp3, rp_record = random_projection_reference(y0, y1, y3, x0.shape[1], ys, smoke=smoke)
        predictions.update(pca_oracle_reference=pca3,
                           random_projection_oracle_reference=rp3)
    method_metrics, field_frames = {}, []
    for method, prediction in predictions.items():
        metrics, fields = detailed_metrics(y3, prediction, scale, target_names)
        method_metrics[method] = metrics
        fields.insert(0, "method", method)
        field_frames.append(fields)
    for start, stop, node in segments:
        for method, prediction in predictions.items():
            subset, _ = detailed_metrics(y3[:, start:stop], prediction[:, start:stop],
                                         scale[start:stop], target_names[start:stop])
            method_metrics[method][f"{node}_macro_nrmse"] = subset["macro_nrmse_all"]

    symbols = folds[3]["keys"].Symbol.astype(str).to_numpy()
    company_index = np.asarray([global_company_to_index[s] for s in symbols], dtype=int)
    recon_reps = reconstruction_bootstrap(y3, chosen3, constant3, scale,
                                          company_index, boot_weights[:BOOT_RECON])
    specs = attribute_specs(target_names, y0, meta, segments)
    reconstruction_attribute_rows = attribute_metrics(y3, chosen3, specs)
    attribute_rows, attribute_scores, attribute_truth3, attribute_fit = \
        fit_direct_attribute_attack(x0, y0, x1, y1, x3, y3, specs, smoke=smoke)
    auc_reps = direct_auc_bootstrap(
        attribute_truth3, attribute_scores, attribute_rows, company_index, boot_weights,
        repeats=(5 if smoke else (20 if primary else BOOT_AUC)))
    point_auc = float(np.mean([r["auc"] for r in attribute_rows])) if attribute_rows else None
    auc_ci = interval(auc_reps, 0.5) if np.isfinite(auc_reps).any() else None
    gain_point = (method_metrics["constant"]["macro_nrmse_all"] -
                  method_metrics[chosen]["macro_nrmse_all"])
    gain_ci = interval(recon_reps, 0.0)
    gain_ci["point_estimate"] = float(gain_point)
    if auc_ci is not None:
        auc_ci["point_estimate"] = point_auc
    statistical_signal = bool(gain_ci["lower_above_null"] or
                              (auc_ci is not None and auc_ci["lower_above_null"]))
    practical_leakage = bool((gain_ci["lower_above_null"] and gain_point >= 0.05) or
                             (auc_ci is not None and auc_ci["lower_above_null"] and
                              point_auc is not None and point_auc >= 0.60))

    train_symbols = set(folds[0]["keys"].Symbol.astype(str))
    seen = np.asarray([s in train_symbols for s in symbols], dtype=bool)
    subgroup_masks = {"seen_company": seen, "unseen_company": ~seen}
    if view == "h2_post":
        # audit::unique_company_year_match is the first appended audit slot and
        # is transmitted as r without scaling.
        r_flag = folds[3]["xs"][2][:, 16] > 0.5
        subgroup_masks.update(audit_match_r1=r_flag, audit_unmatched_r0=~r_flag)
    subgroups = {}
    for label, mask in subgroup_masks.items():
        if int(mask.sum()) < 2:
            continue
        c_nrmse = macro_nrmse(y3[mask], constant3[mask], scale)
        a_nrmse = macro_nrmse(y3[mask], chosen3[mask], scale)
        auc_values = []
        for row in attribute_rows:
            j = int(row["score_column"])
            yy = attribute_truth3[mask, j]
            if yy.min() != yy.max():
                auc_values.append(roc_auc_score(yy, attribute_scores[mask, j]))
        subgroups[label] = {
            "rows": int(mask.sum()), "share": float(mask.mean()),
            "constant_nrmse": float(c_nrmse), "selected_nrmse": float(a_nrmse),
            "gain_vs_constant": float(c_nrmse - a_nrmse),
            "attribute_macro_auc": float(np.mean(auc_values)) if auc_values else None,
            "attribute_fields": int(len(auc_values)),
        }
    record = {
        "year": year, "scope": scope, "view": view,
        "attack_input_dimensions": int(x0.shape[1]), "target_dimensions": int(y0.shape[1]),
        "fold_rows": {"train0": len(x0), "select1": len(x1), "test3": len(x3)},
        "ridge_selection": ridge_selection, "mlp_selection": mlp_record,
        "attacker_selection_nrmse": selection, "selected_attacker": chosen,
        "test_metrics": method_metrics,
        "selected_gain_vs_constant": gain_ci,
        "attribute_macro_auc": auc_ci,
        "attribute_fields_point": attribute_rows,
        "direct_attribute_attack": attribute_fit,
        "reconstruction_attribute_fields_point": reconstruction_attribute_rows,
        "subgroups": subgroups,
        "statistical_recoverability_detected": statistical_signal,
        "practical_leakage_criterion_met": practical_leakage,
        "practical_thresholds": {"nrmse_gain": 0.05, "attribute_auc": 0.60},
        "pca_reference": pca_record, "random_projection_reference": rp_record,
        "elapsed_seconds": time.perf_counter() - started,
        "target_semantics": "preprocessed_model_input",
    }
    stem = f"{year}_{scope}_{view}"
    write_json(output_dir / "records" / f"{stem}.json", record)
    pd.concat(field_frames, ignore_index=True).to_csv(
        output_dir / "fields" / f"{stem}.csv", index=False)
    np.savez_compressed(output_dir / "bootstrap" / f"{stem}.npz",
                        reconstruction_gain=recon_reps, attribute_macro_auc=auc_reps)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--primary", action="store_true",
                        help="Run the decision-relevant deployment views first; omit oracle references.")
    parser.add_argument("--years", default="2020,2021,2022")
    args = parser.parse_args()
    active_years = tuple(int(v) for v in args.years.split(",") if v.strip())
    assert active_years and set(active_years).issubset(set(YEARS))
    torch.set_num_threads(4)
    seed_all(ATTACK_SEED)
    suffix = "smoke" if args.smoke else "formal"
    mode = "primary" if args.primary else "complete"
    output_dir = HERE / f"results/{mode}-audit-v2-{suffix}"
    for folder in ("records", "fields", "bootstrap"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    # Never leave a prior completed manifest visible while endpoints are being added.
    write_json_atomic(output_dir / "manifest.json", {
        "schema_version": 2,
        "status": "running",
        "mode": mode,
        "years": list(active_years),
        "attack_endpoints_rerun_during_rebuild": False,
    })

    # Loading fold 3 here is now authorised; fold 2 is never requested.  The
    # redirected 2022 audit-enriched cache must only be installed after the two
    # earlier packs have been materialised, otherwise it can silently change
    # their loader state as well.
    packs = {year: C.pack_for(year, (0, 1, 3))
             for year in active_years if year in (2020, 2021)}
    if 2022 in active_years:
        import train_audit_year2022 as fix22
        fix22.install_redirected_loaders()
        packs[2022] = C.pack_for(2022, (0, 1, 3))
    folds_by_year = {year: {fold: split for fold, split in zip((0, 1, 3), pack["splits"])}
                     for year, pack in packs.items()}
    all_symbols = sorted(set().union(*[
        set(folds_by_year[year][3]["keys"].Symbol.astype(str)) for year in active_years]))
    company_to_index = {symbol: i for i, symbol in enumerate(all_symbols)}
    rng = np.random.default_rng(ATTACK_SEED)
    boot_weights = rng.poisson(1.0, size=(BOOT_RECON, len(all_symbols))).astype("i2")
    empty = boot_weights.sum(1) == 0
    boot_weights[empty, 0] = 1
    formal = C.shared.load_loader().formal

    records = []
    primary_views = {
        "single42": ("h1", "h2", "h2_post", "h3", "full_deployment"),
        "joint6": ("h1", "h2", "h2_post", "h3", "full_deployment"),
    }
    complete_views = tuple(("h1", "h2", "h2_q", "h2_cr", "h2_qcr", "h2_post", "h3",
                            "full_deployment", "full_implementation"))
    for year in active_years:
        pack, folds = packs[year], folds_by_year[year]
        extracted = {fold: {} for fold in (0, 1, 3)}
        for seed in SEEDS:
            model = A.FilmKeepX3(formal, pack, seed)
            model_path = CONTROL / f"artifacts/{year}/FiLM-L_s{seed}/model.pt"
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            for fold in (0, 1, 3):
                split = folds[fold]
                if args.smoke:
                    split = {**split, "xs": [x[:512] for x in split["xs"]],
                             **{k: split[k][:512] for k in ("q", "state", "industry", "y", "types")}}
                extracted[fold][seed] = extract(model, split)
        views = {fold: build_views(extracted[fold]) for fold in (0, 1, 3)}
        if args.smoke:
            # Preserve key alignment for the smoke subset only.
            folds = {fold: {**folds[fold], "xs": [x[:512] for x in folds[fold]["xs"]],
                            "keys": folds[fold]["keys"].iloc[:512].copy()}
                     for fold in (0, 1, 3)}
        for scope in ("single42", "joint6"):
            for view in (primary_views[scope] if args.primary else complete_views):
                stem = output_dir / "records" / f"{year}_{scope}_{view}.json"
                if stem.exists() and not args.smoke:
                    records.append(json.loads(stem.read_text(encoding="utf-8")))
                    continue
                record = attack_one(year, scope, view, folds, views, pack["names"], pack["meta"],
                                    company_to_index, boot_weights, output_dir,
                                    smoke=args.smoke, primary=args.primary)
                records.append(record)
                print(json.dumps({"year": year, "scope": scope, "view": view,
                                  "selected": record["selected_attacker"],
                                  "gain": record["selected_gain_vs_constant"]["point_estimate"],
                                  "practical_leakage": record.get(
                                      "practical_leakage_criterion_met",
                                      record.get("leakage_criterion_met")),
                                  "seconds": record["elapsed_seconds"]}), flush=True)
        del extracted, views
    yearly, combined, _inventory = build_summaries(
        records, output_dir, active_years, mode)
    write_summaries(output_dir, yearly, combined)
    manifest = build_manifest(
        records=records,
        output_dir=output_dir,
        active_years=active_years,
        model_seeds=SEEDS,
        attack_seed=ATTACK_SEED,
        mode=mode,
        root=ROOT,
        here=HERE,
        control=CONTROL,
        notice=NOTICE,
        builder_path=Path(__file__),
        status="smoke_complete" if args.smoke else "formal_complete",
    )
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
