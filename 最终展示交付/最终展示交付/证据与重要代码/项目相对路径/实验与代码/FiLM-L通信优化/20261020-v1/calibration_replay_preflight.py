"""Rebuild FiLM-L calibration and prove exact fold-3 replay.

This is a preflight utility, not an 81-candidate evaluation.  It reconstructs
the calibration that ``ablation_lib.fit`` kept only in memory, checks that an
FP32 message replay reproduces the checkpoint's raw logits exactly, and then
checks that the reconstructed calibrated logits are exactly the arrays saved in
``fold3.npz``.

The reusable functions ``fit_full_training_scales``, ``replay_messages`` and
``apply_seed_calibration`` are the intended building blocks for a later fixed
precision candidate evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import protocol as wire

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
YEARS = (2020, 2021, 2022)
SEEDS = (42, 100, 17, 223, 483, 907)


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.view(np.uint8))
    return h.hexdigest()


def slices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def load_checkpoint(formal, pack: dict, year: int, seed: int):
    model = wire.A.FilmKeepX3(formal, pack, seed)
    path = wire.CONTROL / "artifacts" / str(year) / f"FiLM-L_s{seed}"
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, path


@torch.no_grad()
def fit_full_training_scales(model, split: dict, batch_size: int = 256):
    """Fit contract-specified per-channel INT8 scales on the whole train fold."""
    maxima: dict[str, np.ndarray] = {}
    for rows in slices(len(split["y"]), batch_size):
        messages = wire.extract_messages(model, split, rows)
        for block, message in messages.items():
            values = message[:, :8] if block == "audit_control" else message
            local = np.max(np.abs(values.astype(np.float64)), axis=0)
            maxima[block] = local if block not in maxima else np.maximum(maxima[block], local)
    return {
        block: np.where(value > 0, value / 127.0, 1.0).astype(np.float32)
        for block, value in maxima.items()
    }


@torch.no_grad()
def replay_messages(model, split: dict, precisions: dict[str, str],
                    scales: dict[str, np.ndarray] | None = None,
                    batch_size: int = 256):
    """Decode selected wire precisions and run the frozen fusion head."""
    scales = scales or {}
    overall, types = [], []
    for rows in slices(len(split["y"]), batch_size):
        messages = wire.extract_messages(model, split, rows)
        decoded = {
            block: wire.quantize_dequantize(
                block, value, precisions[block],
                scales.get(block) if precisions[block] == "int8" else None,
            )
            for block, value in messages.items()
        }
        a, b = wire.fusion_from_messages(model, split["industry"][rows], decoded)
        overall.append(a)
        types.append(b)
    return np.concatenate(overall).astype(np.float64), np.concatenate(types).astype(np.float64)


def rebuild_seed_calibration(formal, model, pack: dict):
    """Repeat the exact fold-2 calibration path used by ``ablation_lib.fit``."""
    a2, b2 = formal.logits(model, pack["splits"][2])
    weights = formal.prev.cost_weights(pack["splits"][0])
    raw = {"2_a_logit": a2, "2_b_logit": b2}
    calibration = wire.C.shared.fit_calibration(formal, raw, pack, weights)
    # The optimizer and inputs must be deterministic before this object is used.
    assert calibration == wire.C.shared.fit_calibration(formal, raw, pack, weights)
    return calibration, weights, a2, b2


def apply_seed_calibration(formal, a: np.ndarray, b: np.ndarray,
                           weights, calibration: dict):
    """Apply the saved-prediction semantics to raw or quantized model logits."""
    a_cal = formal.exp.prob.apply_cal(
        a - np.log(weights[0]), calibration["overall"]
    )
    b_cal = formal.exp.prob.apply_cal(
        b - np.log(weights[1]), calibration["types_shrink"]
    )
    return a_cal, b_cal


def check_one(formal, pack: dict, year: int, seed: int, batch_size: int):
    model, path = load_checkpoint(formal, pack, year, seed)
    run = json.loads((path / "run.json").read_text(encoding="utf8"))
    assert wire.C.shared.digest_state(model) == run["final_hash"]

    calibration, weights, a2, b2 = rebuild_seed_calibration(formal, model, pack)
    direct_a, direct_b = formal.logits(model, pack["splits"][3])
    fp32 = {block: "fp32" for block in wire.BLOCKS}
    replay_a, replay_b = replay_messages(
        model, pack["splits"][3], fp32, batch_size=batch_size
    )
    np.testing.assert_array_equal(replay_a, direct_a)
    np.testing.assert_array_equal(replay_b, direct_b)

    calibrated_a, calibrated_b = apply_seed_calibration(
        formal, replay_a, replay_b, weights, calibration
    )
    with np.load(path / "fold3.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(calibrated_a, saved["a_logit"])
        np.testing.assert_array_equal(calibrated_b, saved["b_logit"])
        saved_a_sha = array_sha256(saved["a_logit"])
        saved_b_sha = array_sha256(saved["b_logit"])

    return {
        "year": year,
        "seed": seed,
        "checkpoint_state_matches_run_hash": True,
        "calibration_file_saved_with_model": (path / "calibration.json").exists(),
        "raw_prediction_file_saved_with_model": (path / "predictions.npz").exists(),
        "fold2_raw_overall_sha256": array_sha256(a2),
        "fold2_raw_types_sha256": array_sha256(b2),
        "fold3_raw_overall_sha256": array_sha256(direct_a),
        "fold3_raw_types_sha256": array_sha256(direct_b),
        "fold3_saved_overall_sha256": saved_a_sha,
        "fold3_saved_types_sha256": saved_b_sha,
        "unquantized_message_replay_raw_logits_exact": True,
        "reconstructed_calibration_matches_fold3_exact": True,
        "weights": {
            "overall": float(weights[0]),
            "types": np.asarray(weights[1]).tolist(),
        },
        "calibration": calibration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS), choices=YEARS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "calibration_replay_preflight.json"
    )
    args = parser.parse_args()
    assert args.batch_size > 0
    torch.set_num_threads(4)

    # The 2022 loader patches process-global module attributes, so it is always last.
    years = [year for year in args.years if year != 2022]
    if 2022 in args.years:
        years.append(2022)
    formal = wire.C.shared.load_loader().formal
    checks = []
    for year in years:
        pack = wire.pack_for(year, range(4))
        for seed in args.seeds:
            checks.append(check_one(formal, pack, year, seed, args.batch_size))
            print(f"PASS year={year} seed={seed}", flush=True)

    assert all(not row["calibration_file_saved_with_model"] for row in checks)
    assert all(not row["raw_prediction_file_saved_with_model"] for row in checks)
    result = {
        "status": "passed",
        "scope": "calibration reconstruction and FP32 replay only; no 81-candidate evaluation",
        "years": years,
        "seeds": args.seeds,
        "runs_checked": len(checks),
        "calibration_parameters_were_saved": False,
        "raw_fold_predictions_were_saved": False,
        "fold3_contains": ["calibrated overall logit", "calibrated conditional-type logits"],
        "exact_reproduction_rule": (
            "reload checkpoint; recompute fold2 raw logits and fold0 cost weights; "
            "refit overall Platt, shared type Platt, and shrink type calibration; "
            "assert FP32 message replay equals direct fold3 raw logits; apply offsets and "
            "the reconstructed per-seed calibration; assert array equality with fold3.npz"
        ),
        "quantized_rule": (
            "fit per-seed/per-block/per-channel INT8 scales on the full training fold; "
            "quantize/dequantize each seed's four messages; fuse with that seed's head; "
            "apply that seed's reconstructed calibration; only then average the six "
            "calibrated logits and apply sigmoid"
        ),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(json.dumps({k: result[k] for k in ("status", "runs_checked", "years", "seeds")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
