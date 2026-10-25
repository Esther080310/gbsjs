"""Deterministic summary and provenance builders for the FiLM-L privacy audit.

This module never loads a model or an attack data split.  It only consumes saved
record JSON files and saved bootstrap arrays.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_SCOPES = ("single42", "joint6")
PRIMARY_VIEWS = ("h1", "h2", "h2_post", "h3", "full_deployment")
COMPLETE_VIEWS = (
    "h1", "h2", "h2_q", "h2_cr", "h2_qcr", "h2_post", "h3",
    "full_deployment", "full_implementation",
)
PRACTICAL_NRMSE_GAIN = 0.05
PRACTICAL_ATTRIBUTE_AUC = 0.60


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".rebuild-tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".rebuild-tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def record_key(record: dict) -> tuple[int, str, str]:
    return int(record["year"]), str(record["scope"]), str(record["view"])


def key_text(key: tuple[int, str, str]) -> str:
    return f"{key[0]}::{key[1]}::{key[2]}"


def expected_record_keys(active_years: Iterable[int], mode: str) -> list[tuple[int, str, str]]:
    years = tuple(int(year) for year in active_years)
    if mode not in ("primary", "complete"):
        raise ValueError(f"unknown audit mode: {mode}")
    views = PRIMARY_VIEWS if mode == "primary" else COMPLETE_VIEWS
    return [(year, scope, view)
            for year in years for scope in PRIMARY_SCOPES for view in views]


def record_inventory(records: list[dict], active_years: Iterable[int], mode: str) -> dict:
    expected_order = expected_record_keys(active_years, mode)
    expected = set(expected_order)
    observed_order = [record_key(record) for record in records]
    counts = Counter(observed_order)
    observed = set(observed_order)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    exact = not duplicates and not missing and not unexpected and len(records) == len(expected)
    return {
        "expected_record_keys": [key_text(key) for key in expected_order],
        "observed_record_keys": [key_text(key) for key in sorted(observed)],
        "missing_record_keys": [key_text(key) for key in missing],
        "unexpected_record_keys": [key_text(key) for key in unexpected],
        "duplicate_record_keys": [key_text(key) for key in duplicates],
        "expected_records": len(expected),
        "observed_records": len(records),
        "all_expected_records": exact,
    }


def criterion_flags(record: dict) -> tuple[bool, bool]:
    gain = record["selected_gain_vs_constant"]
    auc = record.get("attribute_macro_auc")
    gain_point = float(gain["point_estimate"])
    gain_significant = float(gain["ci95"][0]) > 0.0
    auc_point = None if auc is None else auc.get("point_estimate")
    auc_significant = bool(auc is not None and float(auc["ci95"][0]) > 0.5)
    statistical = bool(gain_significant or auc_significant)
    practical = bool(
        (gain_significant and gain_point >= PRACTICAL_NRMSE_GAIN)
        or (auc_significant and auc_point is not None
            and float(auc_point) >= PRACTICAL_ATTRIBUTE_AUC)
    )
    stored_statistical = record.get("statistical_recoverability_detected")
    if stored_statistical is not None and bool(stored_statistical) != statistical:
        raise AssertionError(
            f"stored statistical criterion disagrees for {record_key(record)}")
    stored_practical = record.get(
        "practical_leakage_criterion_met", record.get("leakage_criterion_met"))
    if stored_practical is not None and bool(stored_practical) != practical:
        raise AssertionError(
            f"stored practical criterion disagrees for {record_key(record)}")
    return statistical, practical


def interval(values: np.ndarray, null: float) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        raise ValueError("cannot form an interval from zero finite bootstrap values")
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "estimate": float(np.mean(finite)),
        "ci95": [float(low), float(high)],
        "null": float(null),
        "lower_above_null": bool(low > null),
        "replicates": int(len(finite)),
    }


def build_summaries(records: list[dict], output_dir: Path,
                    active_years: Iterable[int], mode: str) -> tuple[pd.DataFrame, list[dict], dict]:
    years = tuple(int(year) for year in active_years)
    inventory = record_inventory(records, years, mode)
    if not inventory["all_expected_records"]:
        raise AssertionError(json.dumps(inventory, ensure_ascii=False, indent=2))
    lookup = {record_key(record): record for record in records}
    ordered = [lookup[key] for key in expected_record_keys(years, mode)]

    yearly_rows = []
    for record in ordered:
        chosen = record["selected_attacker"]
        statistical, practical = criterion_flags(record)
        auc = record.get("attribute_macro_auc") or {}
        yearly_rows.append({
            "year": record["year"],
            "scope": record["scope"],
            "view": record["view"],
            "attack_dims": record["attack_input_dimensions"],
            "target_dims": record["target_dimensions"],
            "selected_attacker": chosen,
            "constant_nrmse": record["test_metrics"]["constant"]["macro_nrmse_all"],
            "selected_nrmse": record["test_metrics"][chosen]["macro_nrmse_all"],
            "gain_vs_constant": record["selected_gain_vs_constant"]["point_estimate"],
            "gain_ci_low": record["selected_gain_vs_constant"]["ci95"][0],
            "gain_ci_high": record["selected_gain_vs_constant"]["ci95"][1],
            "attribute_macro_auc": auc.get("point_estimate"),
            "auc_ci_low": auc.get("ci95", [None, None])[0],
            "auc_ci_high": auc.get("ci95", [None, None])[1],
            "statistical_recoverability_detected": statistical,
            "practical_leakage_criterion_met": practical,
        })
    yearly = pd.DataFrame(yearly_rows)

    views = PRIMARY_VIEWS if mode == "primary" else COMPLETE_VIEWS
    combined = []
    for scope in PRIMARY_SCOPES:
        for view in views:
            subset = [lookup[(year, scope, view)] for year in years]
            yearly_subset = yearly[(yearly.scope == scope) & (yearly.view == view)]
            gain_replicates, auc_replicates = [], []
            for record in subset:
                stem = f'{record["year"]}_{scope}_{view}'
                with np.load(output_dir / "bootstrap" / f"{stem}.npz") as saved:
                    gain_replicates.append(saved["reconstruction_gain"])
                    auc_replicates.append(saved["attribute_macro_auc"])
            gain_average = np.mean(np.stack(gain_replicates), axis=0)
            auc_average = np.nanmean(np.stack(auc_replicates), axis=0)
            gain_ci = interval(gain_average, 0.0)
            auc_ci = interval(auc_average, 0.5) if np.isfinite(auc_average).any() else None
            gain_point = float(yearly_subset.gain_vs_constant.mean())
            auc_point = (float(yearly_subset.attribute_macro_auc.mean())
                         if yearly_subset.attribute_macro_auc.notna().any() else None)
            statistical = bool(
                gain_ci["lower_above_null"]
                or (auc_ci is not None and auc_ci["lower_above_null"])
            )
            practical = bool(
                (gain_ci["lower_above_null"] and gain_point >= PRACTICAL_NRMSE_GAIN)
                or (auc_ci is not None and auc_ci["lower_above_null"]
                    and auc_point is not None and auc_point >= PRACTICAL_ATTRIBUTE_AUC)
            )
            combined.append({
                "scope": scope,
                "view": view,
                "years": list(years),
                "year_equal_mean_constant_nrmse": float(yearly_subset.constant_nrmse.mean()),
                "year_equal_mean_selected_nrmse": float(yearly_subset.selected_nrmse.mean()),
                "year_equal_mean_gain_point": gain_point,
                "reconstruction_gain_bootstrap": gain_ci,
                "year_equal_mean_attribute_auc_point": auc_point,
                "attribute_auc_bootstrap": auc_ci,
                "statistical_recoverability_detected": statistical,
                "practical_leakage_criterion_met": practical,
            })
    return yearly, combined, inventory


def write_summaries(output_dir: Path, yearly: pd.DataFrame, combined: list[dict]) -> None:
    write_csv_atomic(output_dir / "yearly_summary.csv", yearly)
    write_json_atomic(output_dir / "three_year_summary.json", combined)
    flat = pd.DataFrame([
        {
            **row,
            "gain_ci_low": row["reconstruction_gain_bootstrap"]["ci95"][0],
            "gain_ci_high": row["reconstruction_gain_bootstrap"]["ci95"][1],
            "auc_ci_low": (row["attribute_auc_bootstrap"] or {}).get(
                "ci95", [None, None])[0],
            "auc_ci_high": (row["attribute_auc_bootstrap"] or {}).get(
                "ci95", [None, None])[1],
        }
        for row in combined
    ]).drop(columns=["reconstruction_gain_bootstrap", "attribute_auc_bootstrap"])
    write_csv_atomic(output_dir / "three_year_summary.csv", flat)


def _hash_existing(paths: Iterable[Path]) -> dict[str, str]:
    paths = [Path(path).resolve() for path in paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing provenance files: " + ", ".join(missing))
    return {str(path): digest(path) for path in paths}


def provenance_paths(root: Path, here: Path, control: Path, notice: Path,
                     active_years: Iterable[int], model_seeds: Iterable[int],
                     builder_path: Path) -> dict:
    years = tuple(int(year) for year in active_years)
    seeds = tuple(int(seed) for seed in model_seeds)
    source_paths = [
        here / "训练前约定.md",
        here / "run_formal_audit.py",
        here / "audit_artifacts.py",
        builder_path,
        control / "ablation_lib.py",
        control / "cma_lib.py",
        notice / "scripts/train_auxiliary_semantic.py",
        notice / "scripts/train_audit_information.py",
        notice / "scripts/train_audit_year2022.py",
        notice / "notice_pipeline/audit_information.py",
        root / "实验与代码/财务年度配对修正/pipeline.py",
    ]
    source_paths = list(dict.fromkeys(Path(path).resolve() for path in source_paths))
    model_paths = [
        control / f"artifacts/{year}/FiLM-L_s{seed}/model.pt"
        for year in years for seed in seeds
    ]
    data_paths = []
    data_index = {}
    old_features = (notice /
        "artifacts/experiments/20261016-evidence-v1/audit_information/features")
    new_features = notice / "artifacts/experiments/20261016-audit-year2022-v1/features"
    new_cache = notice / "artifacts/experiments/20261016-audit-year2022-v1/cache"
    for year in years:
        if year in (2020, 2021):
            base = root / f"实验与代码/财务年度配对修正/cache/{year}"
            year_paths = [base.with_suffix(".npz"), base.with_suffix(".json"),
                          old_features / f"audit_features_{year}.npz"]
            feature_manifest = old_features / "manifest.json"
        elif year == 2022:
            year_paths = [new_cache / "2022.npz", new_cache / "2022.json",
                          new_features / "audit_features_2022.npz"]
            feature_manifest = new_features / "manifest.json"
        else:
            raise ValueError(f"no frozen data-pack index registered for year {year}")
        year_paths.append(feature_manifest)
        data_index[str(year)] = {
            "base_pack_npz": str(year_paths[0].resolve()),
            "base_pack_metadata": str(year_paths[1].resolve()),
            "audit_features": str(year_paths[2].resolve()),
            "audit_feature_manifest": str(year_paths[3].resolve()),
        }
        data_paths.extend(year_paths)
    data_paths = list(dict.fromkeys(Path(path).resolve() for path in data_paths))
    return {
        "source_sha256": _hash_existing(source_paths),
        "model_sha256": _hash_existing(model_paths),
        "data_pack_sha256": _hash_existing(data_paths),
        "data_pack_index": data_index,
    }


def build_manifest(records: list[dict], output_dir: Path, active_years: Iterable[int],
                   model_seeds: Iterable[int], attack_seed: int, mode: str,
                   root: Path, here: Path, control: Path, notice: Path,
                   builder_path: Path, status: str = "formal_complete",
                   previous_manifest: dict | None = None) -> dict:
    years = tuple(int(year) for year in active_years)
    seeds = tuple(int(seed) for seed in model_seeds)
    inventory = record_inventory(records, years, mode)
    if not inventory["all_expected_records"]:
        raise AssertionError(json.dumps(inventory, ensure_ascii=False, indent=2))
    flags = [criterion_flags(record) for record in records]
    provenance = provenance_paths(root, here, control, notice, years, seeds, builder_path)
    record_files = [output_dir / "records" / f"{key[0]}_{key[1]}_{key[2]}.json"
                    for key in expected_record_keys(years, mode)]
    bootstrap_files = [output_dir / "bootstrap" / f"{key[0]}_{key[1]}_{key[2]}.npz"
                       for key in expected_record_keys(years, mode)]
    summary_files = [output_dir / "yearly_summary.csv",
                     output_dir / "three_year_summary.csv",
                     output_dir / "three_year_summary.json"]
    manifest = {
        "schema_version": 2,
        "status": status,
        "mode": mode,
        "fold_contract": {"train": 0, "selection": 1, "unused": 2, "formal_test": 3},
        "target_semantics": "preprocessed_model_input_not_raw_csmar",
        "years": list(years),
        "model_seeds": list(seeds),
        "attack_seed": int(attack_seed),
        "records": len(records),
        **inventory,
        "statistical_recoverability_records": int(sum(statistical for statistical, _ in flags)),
        "practical_leakage_records": int(sum(practical for _, practical in flags)),
        "criterion_definitions": {
            "statistical_recoverability_detected": (
                "reconstruction gain CI95 lower > 0 OR attribute macro-AUC CI95 lower > 0.5"),
            "practical_leakage_criterion_met": (
                "statistical reconstruction signal with point gain >= 0.05 OR statistical "
                "attribute signal with point macro-AUC >= 0.60"),
            "nrmse_gain_threshold": PRACTICAL_NRMSE_GAIN,
            "attribute_auc_threshold": PRACTICAL_ATTRIBUTE_AUC,
        },
        **provenance,
        "record_sha256": _hash_existing(record_files),
        "bootstrap_sha256": _hash_existing(bootstrap_files),
        "summary_sha256": _hash_existing(summary_files),
        "artifact_builder": str(builder_path.resolve()),
        "attack_endpoints_rerun_during_rebuild": False,
    }
    if previous_manifest is not None:
        historical = previous_manifest.get("record_generation_manifest_before_migration")
        if historical is None:
            historical = {
                "status": previous_manifest.get("status"),
                "years": previous_manifest.get("years"),
                "records": previous_manifest.get("records"),
                "source_sha256": previous_manifest.get("source_sha256", {}),
            }
        manifest["record_generation_manifest_before_migration"] = historical
    return manifest
