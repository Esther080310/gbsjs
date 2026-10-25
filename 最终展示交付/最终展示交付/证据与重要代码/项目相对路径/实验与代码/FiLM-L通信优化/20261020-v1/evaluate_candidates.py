"""Evaluate frozen FiLM-L message-precision policies.

The default command is deliberately limited to an all-FP32 identity check and
one all-FP16, label-free smoke test.  Running the full 81-policy matrix requires
both ``--run-full-matrix`` and an approved privacy-interface contract.  The
program never ranks or selects a candidate automatically.

Frozen prediction order for every policy:

    seed messages -> quantize/dequantize -> seed fusion raw logits
    -> seed-specific calibration -> mean of six calibrated logits -> sigmoid
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import calibration_replay_preflight as replay
import protocol as wire

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONTRACT_PATH = HERE / "protocol_contract.json"
CANDIDATE_PATH = RESULTS / "candidate_manifest.json"
YEARS = (2020, 2021, 2022)
SEEDS = (42, 100, 17, 223, 483, 907)
ALL_FP32 = {block: "fp32" for block in wire.BLOCKS}
ALL_FP16 = {block: "fp16" for block in wire.BLOCKS}


@dataclass
class SeedRuntime:
    year: int
    seed: int
    model: Any
    messages: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    weights: Any
    calibration: dict
    saved_a_logit: np.ndarray
    saved_b_logit: np.ndarray


@dataclass
class YearRuntime:
    year: int
    split: dict
    seeds: list[SeedRuntime]
    baseline_a_logit: np.ndarray
    baseline_b_logit: np.ndarray
    baseline_a: np.ndarray
    baseline_b: np.ndarray
    baseline_selected: np.ndarray
    baseline_top2: np.ndarray


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    out = np.empty_like(value)
    nonnegative = value >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exp_value = np.exp(value[~nonnegative])
    out[~nonnegative] = exp_value / (1.0 + exp_value)
    return out


def selected_mask(score: np.ndarray) -> np.ndarray:
    k = int(np.ceil(0.1 * len(score)))
    selected = np.zeros(len(score), dtype=bool)
    selected[np.argsort(-score, kind="stable")[:k]] = True
    return selected


def top2_mask(score: np.ndarray) -> np.ndarray:
    top = np.argsort(-score, axis=1, kind="stable")[:, :2]
    mask = np.zeros(score.shape, dtype=bool)
    mask[np.arange(len(score))[:, None], top] = True
    assert np.all(mask.sum(axis=1) == 2)
    return mask


@torch.no_grad()
def extract_all_messages(model, split: dict, batch_size: int) -> dict[str, np.ndarray]:
    pieces = {block: [] for block in wire.BLOCKS}
    for rows in replay.slices(len(split["industry"]), batch_size):
        current = wire.extract_messages(model, split, rows)
        for block in wire.BLOCKS:
            pieces[block].append(current[block])
    return {block: np.concatenate(values) for block, values in pieces.items()}


@torch.no_grad()
def fuse_seed(seed: SeedRuntime, split: dict, policy: dict[str, str], batch_size: int):
    overall, types = [], []
    for rows in replay.slices(len(split["industry"]), batch_size):
        decoded = {
            block: wire.quantize_dequantize(
                block,
                seed.messages[block][rows],
                policy[block],
                seed.scales[block] if policy[block] == "int8" else None,
            )
            for block in wire.BLOCKS
        }
        a, b = wire.fusion_from_messages(seed.model, split["industry"][rows], decoded)
        overall.append(a)
        types.append(b)
    return np.concatenate(overall).astype(np.float64), np.concatenate(types).astype(np.float64)


def prepare_runtime(batch_size: int, scale_path: Path, scale_manifest_path: Path):
    """Load frozen models, rebuild calibration, and fit train-only INT8 scales."""
    started = time.perf_counter()
    formal = wire.C.shared.load_loader().formal
    runtime: dict[int, YearRuntime] = {}
    scale_arrays: dict[str, np.ndarray] = {}
    scale_records: list[dict] = []
    source_records: list[dict] = []

    # Installing the redirected 2022 loader mutates process-global module state.
    # Keep it last so the 2020/2021 packs cannot accidentally use the 2022 cache.
    for year in YEARS:
        pack = wire.pack_for(year, range(4))
        split = pack["splits"][3]
        seed_runs: list[SeedRuntime] = []
        for seed in SEEDS:
            model, artifact = replay.load_checkpoint(formal, pack, year, seed)
            run = json.loads((artifact / "run.json").read_text(encoding="utf8"))
            assert wire.C.shared.digest_state(model) == run["final_hash"]
            calibration, weights, _a2, _b2 = replay.rebuild_seed_calibration(
                formal, model, pack
            )
            scales = replay.fit_full_training_scales(
                model, pack["splits"][0], batch_size=batch_size
            )
            messages = extract_all_messages(model, split, batch_size)
            with np.load(artifact / "fold3.npz", allow_pickle=False) as saved:
                saved_a = saved["a_logit"].copy()
                saved_b = saved["b_logit"].copy()
            assert saved_a.shape == (len(split["industry"]),)
            assert saved_b.shape == (len(split["industry"]), 10)
            seed_runs.append(SeedRuntime(
                year=year,
                seed=seed,
                model=model,
                messages=messages,
                scales=scales,
                weights=weights,
                calibration=calibration,
                saved_a_logit=saved_a,
                saved_b_logit=saved_b,
            ))
            for block in wire.BLOCKS:
                key = f"y{year}_s{seed}_{block}"
                scale_arrays[key] = scales[block]
                scale_records.append({
                    "year": year,
                    "seed": seed,
                    "block": block,
                    "key": key,
                    "channels": int(len(scales[block])),
                    "dtype": str(scales[block].dtype),
                    "sha256": replay.array_sha256(scales[block]),
                    "minimum": float(scales[block].min()),
                    "maximum": float(scales[block].max()),
                    "fit_rows": int(len(pack["splits"][0]["industry"])),
                    "fit_fold": 0,
                })
            source_records.append({
                "year": year,
                "seed": seed,
                "model_path": str(artifact / "model.pt"),
                "model_sha256": sha256_file(artifact / "model.pt"),
                "fold3_path": str(artifact / "fold3.npz"),
                "fold3_sha256": sha256_file(artifact / "fold3.npz"),
                "run_final_hash": run["final_hash"],
            })

        # q1/q2/q3 and r are common inputs.  Their scales/decoded values can be
        # shared in the six-model wire ledger without changing model outputs.
        for block in ("market", "finance", "governance"):
            reference_value = seed_runs[0].messages[block][:, -1]
            reference_scale = seed_runs[0].scales[block][-1]
            for seed_run in seed_runs[1:]:
                np.testing.assert_array_equal(reference_value, seed_run.messages[block][:, -1])
                np.testing.assert_array_equal(reference_scale, seed_run.scales[block][-1])
        reference_r = seed_runs[0].messages["audit_control"][:, -1]
        for seed_run in seed_runs[1:]:
            np.testing.assert_array_equal(reference_r, seed_run.messages["audit_control"][:, -1])
        assert set(np.unique(reference_r)).issubset({0.0, 1.0})

        a_seed = np.stack([item.saved_a_logit for item in seed_runs])
        b_seed = np.stack([item.saved_b_logit for item in seed_runs])
        baseline_a_logit = np.mean(a_seed, axis=0)
        baseline_b_logit = np.mean(b_seed, axis=0)
        baseline_a = sigmoid(baseline_a_logit)
        baseline_b = sigmoid(baseline_b_logit)
        runtime[year] = YearRuntime(
            year=year,
            split=split,
            seeds=seed_runs,
            baseline_a_logit=baseline_a_logit,
            baseline_b_logit=baseline_b_logit,
            baseline_a=baseline_a,
            baseline_b=baseline_b,
            baseline_selected=selected_mask(baseline_a),
            baseline_top2=top2_mask(baseline_b),
        )

    scale_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scale_path, **scale_arrays)
    manifest = {
        "status": "frozen train-fold scales; no candidate selection",
        "fit_rule": "per year, per seed, per logical block, per channel; full training fold absolute maximum / 127",
        "evaluation_rows_used": False,
        "labels_used_for_scale_fit": False,
        "audit_match_r_scaled": False,
        "q_and_r_common_across_seeds_verified": True,
        "npz_path": str(scale_path),
        "npz_sha256": sha256_file(scale_path),
        "records": scale_records,
        "sources": source_records,
    }
    write_json(scale_manifest_path, manifest)
    return formal, runtime, manifest, time.perf_counter() - started


def conditional_macro_ap(y: np.ndarray, types: np.ndarray, score: np.ndarray):
    positive = y == 1
    values = []
    for column in range(types.shape[1]):
        target = types[positive, column]
        if 0 < target.sum() < len(target):
            values.append(float(average_precision_score(target, score[positive, column])))
    return float(np.mean(values)) if values else None


def label_metrics(split: dict, a: np.ndarray, b: np.ndarray, selected: np.ndarray):
    y = split["y"]
    types = split["types"]
    positives = int(y.sum())
    found = int(y[selected].sum())
    return {
        "found": found,
        "recall_at_10_percent": float(found / positives),
        "overall_ap": float(average_precision_score(y, a)),
        "conditional_type_macro_ap": conditional_macro_ap(y, types, b),
    }


def year_comparison(year_run: YearRuntime, a_logit: np.ndarray, b_logit: np.ndarray,
                    contract: dict, include_labels: bool):
    a = sigmoid(a_logit)
    b = sigmoid(b_logit)
    selected = selected_mask(a)
    top2 = top2_mask(b)
    base_selected = year_run.baseline_selected
    base_top2 = year_run.baseline_top2
    intersection = int(np.logical_and(selected, base_selected).sum())
    union = int(np.logical_or(selected, base_selected).sum())
    top2_row_equal = np.all(top2 == base_top2, axis=1)
    overall_error = float(np.max(np.abs(a - year_run.baseline_a)))
    type_error = float(np.max(np.abs(b - year_run.baseline_b)))
    limits = contract["main_all_required"]
    exact_list = bool(np.array_equal(selected, base_selected))
    exact_top2 = bool(np.all(top2_row_equal[base_selected]))
    overall_within = bool(
        overall_error <= limits["max_overall_probability_absolute_error"]
    )
    type_within = bool(type_error <= limits["max_type_probability_absolute_error"])
    strict = {
        "top10_company_set_exact": exact_list,
        "top2_type_set_exact_on_baseline_top10": exact_top2,
        "max_overall_probability_absolute_error": overall_error,
        "max_type_probability_absolute_error": type_error,
        "overall_probability_error_within_0_005": overall_within,
        "type_probability_error_within_0_005": type_within,
        "all_required_pass": bool(exact_list and exact_top2 and overall_within and type_within),
    }
    top10_jaccard = float(intersection / union)
    top2_agreement = float(top2_row_equal[base_selected].mean())
    secondary_limits = contract["secondary_sensitivity_only"]
    secondary = {
        "top10_jaccard": top10_jaccard,
        "top2_set_agreement_on_baseline_top10": top2_agreement,
        "top2_set_agreement_all_rows": float(top2_row_equal.mean()),
        "top10_changed_members_symmetric_difference": int(
            np.logical_xor(selected, base_selected).sum()
        ),
        "top10_jaccard_at_least_0_995": bool(
            top10_jaccard >= secondary_limits["minimum_top10_jaccard_each_year"]
        ),
        "top2_agreement_at_least_0_995": bool(
            top2_agreement >= secondary_limits["minimum_top2_set_agreement_each_year"]
        ),
        "recall_at_10_percent_drop_percentage_points": None,
        "recall_drop_at_most_0_5pp": None,
        "all_secondary_thresholds_pass": None,
        "may_replace_main_decision": False,
    }
    supplementary = None
    if include_labels:
        baseline_metrics = label_metrics(
            year_run.split,
            year_run.baseline_a,
            year_run.baseline_b,
            base_selected,
        )
        candidate_metrics = label_metrics(year_run.split, a, b, selected)
        recall_drop = 100.0 * (
            baseline_metrics["recall_at_10_percent"]
            - candidate_metrics["recall_at_10_percent"]
        )
        recall_ok = bool(
            recall_drop
            <= secondary_limits["maximum_recall_at_10_percent_drop_percentage_points_each_year"]
        )
        secondary["recall_at_10_percent_drop_percentage_points"] = float(recall_drop)
        secondary["recall_drop_at_most_0_5pp"] = recall_ok
        secondary["all_secondary_thresholds_pass"] = bool(
            secondary["top10_jaccard_at_least_0_995"]
            and secondary["top2_agreement_at_least_0_995"]
            and recall_ok
        )
        supplementary = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "overall_ap_difference": float(
                candidate_metrics["overall_ap"] - baseline_metrics["overall_ap"]
            ),
            "conditional_type_macro_ap_difference": float(
                candidate_metrics["conditional_type_macro_ap"]
                - baseline_metrics["conditional_type_macro_ap"]
            ),
        }
    return {
        "year": year_run.year,
        "rows": int(len(a)),
        "top10_budget": int(base_selected.sum()),
        "strict": strict,
        "secondary": secondary,
        "supplementary_label_metrics": supplementary,
        "candidate_a_logit_sha256": replay.array_sha256(a_logit),
        "candidate_b_logit_sha256": replay.array_sha256(b_logit),
    }


def byte_ledger(candidate: dict, rows_by_year: dict[int, int], batch_size: int):
    ensemble = candidate["six_model_logit_ensemble_shared_q_r"]
    single = candidate["single_model"]
    per_year = []
    for year in YEARS:
        rows = rows_by_year[year]
        batches = int(math.ceil(rows / batch_size))
        payload = int(rows * ensemble["payload_bytes_per_company"])
        headers = int(batches * 4 * wire.HEADER.size)
        scales = int(ensemble["one_time_int8_scale_manifest_bytes"])
        per_year.append({
            "year": year,
            "rows": rows,
            "batches": batches,
            "payload_bytes": payload,
            "four_frame_header_bytes": headers,
            "one_time_scale_manifest_bytes": scales,
            "inference_wire_bytes_excluding_scale_manifest": payload + headers,
            "inference_plus_one_time_scale_bytes": payload + headers + scales,
        })
    return {
        "single_model": single,
        "six_model_shared_q_r": ensemble,
        "per_year": per_year,
        "three_year_rows": int(sum(rows_by_year.values())),
        "three_year_payload_bytes": int(sum(row["payload_bytes"] for row in per_year)),
        "three_year_header_bytes": int(sum(row["four_frame_header_bytes"] for row in per_year)),
        "three_year_one_time_scale_manifest_bytes": int(
            sum(row["one_time_scale_manifest_bytes"] for row in per_year)
        ),
        "three_year_inference_plus_scale_bytes": int(
            sum(row["inference_plus_one_time_scale_bytes"] for row in per_year)
        ),
        "physical_data_nodes": 3,
        "logical_frames_per_batch": 4,
        "transport_rounds_per_batch": 1,
        "downlink_bytes": 0,
        "excluded": ["request/acknowledgement", "TCP/TLS", "encryption expansion"],
    }


def three_year_summary(years: list[dict], include_labels: bool):
    strict = {
        "all_years_top10_exact": bool(
            all(row["strict"]["top10_company_set_exact"] for row in years)
        ),
        "all_years_top2_exact": bool(
            all(row["strict"]["top2_type_set_exact_on_baseline_top10"] for row in years)
        ),
        "max_overall_probability_absolute_error": float(max(
            row["strict"]["max_overall_probability_absolute_error"] for row in years
        )),
        "max_type_probability_absolute_error": float(max(
            row["strict"]["max_type_probability_absolute_error"] for row in years
        )),
        "all_required_pass_all_years": bool(
            all(row["strict"]["all_required_pass"] for row in years)
        ),
    }
    secondary = {
        "minimum_top10_jaccard": float(min(
            row["secondary"]["top10_jaccard"] for row in years
        )),
        "minimum_top2_set_agreement_on_baseline_top10": float(min(
            row["secondary"]["top2_set_agreement_on_baseline_top10"] for row in years
        )),
        "maximum_recall_drop_percentage_points": None,
        "all_secondary_thresholds_pass_all_years": None,
        "may_replace_main_decision": False,
    }
    supplementary = None
    if include_labels:
        secondary["maximum_recall_drop_percentage_points"] = float(max(
            row["secondary"]["recall_at_10_percent_drop_percentage_points"]
            for row in years
        ))
        secondary["all_secondary_thresholds_pass_all_years"] = bool(all(
            row["secondary"]["all_secondary_thresholds_pass"] for row in years
        ))
        supplementary = {}
        for name in ("overall_ap", "conditional_type_macro_ap", "recall_at_10_percent"):
            supplementary[f"baseline_equal_year_mean_{name}"] = float(np.mean([
                row["supplementary_label_metrics"]["baseline"][name] for row in years
            ]))
            supplementary[f"candidate_equal_year_mean_{name}"] = float(np.mean([
                row["supplementary_label_metrics"]["candidate"][name] for row in years
            ]))
    return {
        "strict": strict,
        "secondary": secondary,
        "supplementary_label_metrics": supplementary,
    }


def evaluate_candidate(formal, runtime: dict[int, YearRuntime], candidate: dict,
                       contract: dict, include_labels: bool, batch_size: int):
    policy = candidate["precision_by_logical_block"]
    is_fp32 = policy == ALL_FP32
    started = time.perf_counter()
    year_rows = []
    seed_checks = []
    for year in YEARS:
        year_run = runtime[year]
        calibrated_a, calibrated_b = [], []
        for seed_run in year_run.seeds:
            raw_a, raw_b = fuse_seed(seed_run, year_run.split, policy, batch_size)
            a_seed, b_seed = replay.apply_seed_calibration(
                formal,
                raw_a,
                raw_b,
                seed_run.weights,
                seed_run.calibration,
            )
            if is_fp32:
                np.testing.assert_array_equal(a_seed, seed_run.saved_a_logit)
                np.testing.assert_array_equal(b_seed, seed_run.saved_b_logit)
            calibrated_a.append(a_seed)
            calibrated_b.append(b_seed)
            seed_checks.append({
                "year": year,
                "seed": seed_run.seed,
                "fp32_matches_saved_fold3_exact": True if is_fp32 else None,
                "calibrated_a_sha256": replay.array_sha256(a_seed),
                "calibrated_b_sha256": replay.array_sha256(b_seed),
            })
        a_logit = np.mean(np.stack(calibrated_a), axis=0)
        b_logit = np.mean(np.stack(calibrated_b), axis=0)
        if is_fp32:
            np.testing.assert_array_equal(a_logit, year_run.baseline_a_logit)
            np.testing.assert_array_equal(b_logit, year_run.baseline_b_logit)
        year_rows.append(year_comparison(
            year_run, a_logit, b_logit, contract, include_labels
        ))
    rows_by_year = {year: int(len(runtime[year].split["industry"])) for year in YEARS}
    return {
        "candidate_number": candidate["candidate_number"],
        "candidate_id": candidate["candidate_id"],
        "precision_by_logical_block": policy,
        "evaluation_status": (
            "formal_no_automatic_selection" if include_labels else "label_free_smoke_only"
        ),
        "labels_read_for_metrics": include_labels,
        "fp32_seed_and_ensemble_identity_exact": True if is_fp32 else None,
        "years": year_rows,
        "three_year": three_year_summary(year_rows, include_labels),
        "bytes": byte_ledger(candidate, rows_by_year, batch_size),
        "quantize_fuse_calibrate_seconds": time.perf_counter() - started,
        "seed_output_checks": seed_checks,
    }


def flatten(prefix: str, value: Any, out: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, item, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = json.dumps(value, ensure_ascii=False, allow_nan=False)
    else:
        out[prefix] = value


def write_full_csvs(candidates: list[dict]) -> None:
    year_rows, three_rows, byte_rows = [], [], []
    for candidate in candidates:
        base = {
            "candidate_number": candidate["candidate_number"],
            "candidate_id": candidate["candidate_id"],
        }
        for year in candidate["years"]:
            row = dict(base)
            flatten("", year, row)
            year_rows.append(row)
        row = dict(base)
        flatten("", candidate["three_year"], row)
        three_rows.append(row)
        row = dict(base)
        flatten("", candidate["bytes"], row)
        byte_rows.append(row)
    pd.DataFrame(year_rows).to_csv(RESULTS / "formal_candidate_year.csv", index=False)
    pd.DataFrame(three_rows).to_csv(RESULTS / "formal_candidate_three_year.csv", index=False)
    pd.DataFrame(byte_rows).to_csv(RESULTS / "formal_candidate_bytes.csv", index=False)


def load_privacy_approval(path: Path | None) -> dict:
    if path is None or not path.exists():
        raise SystemExit(
            "Full matrix blocked: provide --privacy-interface-contract after the "
            "privacy boundary is approved."
        )
    value = json.loads(path.read_text(encoding="utf8"))
    if value.get("allow_formal_communication_evaluation") is not True:
        raise SystemExit(
            "Full matrix blocked: privacy contract must set "
            "allow_formal_communication_evaluation=true."
        )
    return {"path": str(path), "sha256": sha256_file(path), "contract": value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--run-full-matrix", action="store_true")
    parser.add_argument("--privacy-interface-contract", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to evaluator_smoke.json or formal_candidate_evaluation.json.",
    )
    args = parser.parse_args()
    assert args.batch_size > 0
    torch.set_num_threads(4)

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf8"))
    manifest = json.loads(CANDIDATE_PATH.read_text(encoding="utf8"))
    assert manifest["count"] == 81 and len(manifest["candidates"]) == 81
    assert contract["candidate_grid"]["count"] == 81
    assert contract["ensemble"]["aggregation"] == "equal arithmetic mean of six calibrated logits"
    assert contract["ensemble"]["probability_transform"] == "sigmoid after logit averaging"

    privacy = None
    if args.run_full_matrix:
        privacy = load_privacy_approval(args.privacy_interface_contract)
        candidates = manifest["candidates"]
        include_labels = True
        default_output = RESULTS / "formal_candidate_evaluation.json"
    else:
        assert args.privacy_interface_contract is None, (
            "A privacy contract is only consumed with --run-full-matrix."
        )
        candidates = [
            next(row for row in manifest["candidates"]
                 if row["precision_by_logical_block"] == ALL_FP32),
            next(row for row in manifest["candidates"]
                 if row["precision_by_logical_block"] == ALL_FP16),
        ]
        include_labels = False
        default_output = RESULTS / "evaluator_smoke.json"

    formal, runtime, scale_manifest, preparation_seconds = prepare_runtime(
        args.batch_size,
        RESULTS / "int8_scales_training_fold.npz",
        RESULTS / "int8_scale_manifest.json",
    )
    evaluations = []
    for candidate in candidates:
        evaluations.append(evaluate_candidate(
            formal, runtime, candidate, contract, include_labels, args.batch_size
        ))
        print(f"PASS {candidate['candidate_id']}", flush=True)

    if not args.run_full_matrix:
        assert len(evaluations) == 2
        assert evaluations[0]["precision_by_logical_block"] == ALL_FP32
        assert evaluations[0]["fp32_seed_and_ensemble_identity_exact"] is True
        assert evaluations[1]["precision_by_logical_block"] == ALL_FP16
        assert all(row["supplementary_label_metrics"] is None
                   for result in evaluations for row in result["years"])

    output = {
        "status": (
            "full 81-policy evaluation complete; no automatic selection"
            if args.run_full_matrix
            else "preflight passed: FP32 identity plus one FP16 label-free smoke; full matrix not run"
        ),
        "run_full_matrix": args.run_full_matrix,
        "candidate_count_evaluated": len(evaluations),
        "candidate_count_available": 81,
        "labels_read_for_metrics": include_labels,
        "privacy_interface": privacy,
        "frozen_prediction_order": (
            "per-seed messages -> quantize/dequantize -> per-seed fusion raw logits "
            "-> per-seed calibration -> equal mean of six calibrated logits -> sigmoid"
        ),
        "scale_manifest": {
            "path": str(RESULTS / "int8_scale_manifest.json"),
            "sha256": sha256_file(RESULTS / "int8_scale_manifest.json"),
            "npz_path": scale_manifest["npz_path"],
            "npz_sha256": scale_manifest["npz_sha256"],
            "fit_rule": scale_manifest["fit_rule"],
        },
        "protocol_contract": {
            "path": str(CONTRACT_PATH),
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "candidate_manifest": {
            "path": str(CANDIDATE_PATH),
            "sha256": sha256_file(CANDIDATE_PATH),
        },
        "batch_size": args.batch_size,
        "preparation_seconds": preparation_seconds,
        "evaluations": evaluations,
    }
    target = args.output or default_output
    write_json(target, output)
    if args.run_full_matrix:
        write_full_csvs(evaluations)
    print(json.dumps({
        "status": output["status"],
        "evaluated": len(evaluations),
        "output": str(target),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
