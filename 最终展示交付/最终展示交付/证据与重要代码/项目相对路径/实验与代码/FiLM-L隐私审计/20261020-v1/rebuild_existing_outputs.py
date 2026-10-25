"""Rebuild summaries and provenance from saved FiLM-L audit records.

No model, data split, or attack is loaded.  The script reads record JSON files and
bootstrap NPZ files, verifies that all expected endpoint keys exist exactly once,
and atomically replaces the two summaries and manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from audit_artifacts import (
    build_manifest,
    build_summaries,
    record_key,
    write_json_atomic,
    write_summaries,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTROL = ROOT / "实验与代码/控制消息架构/20261018-v1"
NOTICE = ROOT / "实验与代码/违规通报结构化研究"
DEFAULT_OUTPUT = HERE / "results/primary-audit-v2-formal"
MODEL_SEEDS = (42, 100, 17, 223, 483, 907)
ATTACK_SEED = 20261020

YEARLY_NUMERIC_FIELDS = (
    "constant_nrmse", "selected_nrmse", "gain_vs_constant",
    "gain_ci_low", "gain_ci_high", "attribute_macro_auc",
    "auc_ci_low", "auc_ci_high",
)
THREE_YEAR_NUMERIC_FIELDS = (
    "year_equal_mean_constant_nrmse",
    "year_equal_mean_selected_nrmse",
    "year_equal_mean_gain_point",
    "year_equal_mean_attribute_auc_point",
)


def load_records(output_dir: Path) -> list[dict]:
    records = []
    for path in sorted((output_dir / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        expected_name = f"{record['year']}_{record['scope']}_{record['view']}.json"
        if path.name != expected_name:
            raise AssertionError(f"record filename/content mismatch: {path.name} != {expected_name}")
        records.append(record)
    return records


def old_numeric_snapshot(output_dir: Path) -> dict:
    with (output_dir / "yearly_summary.csv").open(newline="", encoding="utf-8") as handle:
        yearly = {
            (int(row["year"]), row["scope"], row["view"]):
            {field: float(row[field]) for field in YEARLY_NUMERIC_FIELDS}
            for row in csv.DictReader(handle)
        }
    three_rows = json.loads((output_dir / "three_year_summary.json").read_text(encoding="utf-8"))
    three = {}
    for row in three_rows:
        item = {field: float(row[field]) for field in THREE_YEAR_NUMERIC_FIELDS}
        item.update(
            gain_ci_low=float(row["reconstruction_gain_bootstrap"]["ci95"][0]),
            gain_ci_high=float(row["reconstruction_gain_bootstrap"]["ci95"][1]),
            auc_ci_low=float(row["attribute_auc_bootstrap"]["ci95"][0]),
            auc_ci_high=float(row["attribute_auc_bootstrap"]["ci95"][1]),
        )
        three[(row["scope"], row["view"])] = item
    return {"yearly": yearly, "three_year": three}


def new_numeric_snapshot(yearly, combined: list[dict]) -> dict:
    yearly_values = {}
    for row in yearly.to_dict("records"):
        key = (int(row["year"]), row["scope"], row["view"])
        yearly_values[key] = {field: float(row[field]) for field in YEARLY_NUMERIC_FIELDS}
    three_values = {}
    for row in combined:
        item = {field: float(row[field]) for field in THREE_YEAR_NUMERIC_FIELDS}
        item.update(
            gain_ci_low=float(row["reconstruction_gain_bootstrap"]["ci95"][0]),
            gain_ci_high=float(row["reconstruction_gain_bootstrap"]["ci95"][1]),
            auc_ci_low=float(row["attribute_auc_bootstrap"]["ci95"][0]),
            auc_ci_high=float(row["attribute_auc_bootstrap"]["ci95"][1]),
        )
        three_values[(row["scope"], row["view"])] = item
    return {"yearly": yearly_values, "three_year": three_values}


def assert_numeric_unchanged(old: dict, new: dict) -> None:
    if old.keys() != new.keys():
        raise AssertionError(f"numeric summary sections changed: {old.keys()} != {new.keys()}")
    for section in old:
        if old[section].keys() != new[section].keys():
            raise AssertionError(f"{section} endpoint keys changed")
        for key in old[section]:
            for field, old_value in old[section][key].items():
                new_value = new[section][key][field]
                if old_value != new_value:
                    raise AssertionError(
                        f"numeric value changed at {section}/{key}/{field}: "
                        f"{old_value!r} != {new_value!r}")


def snapshot_digest(snapshot: dict) -> str:
    serializable = {
        section: {"|".join(map(str, key)): values
                  for key, values in sorted(rows.items())}
        for section, rows in snapshot.items()
    }
    payload = json.dumps(serializable, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--years", default="2020,2021,2022")
    parser.add_argument("--mode", choices=("primary", "complete"), default="primary")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    active_years = tuple(int(value) for value in args.years.split(",") if value.strip())
    previous_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    before = old_numeric_snapshot(output_dir)
    records = load_records(output_dir)
    yearly, combined, inventory = build_summaries(
        records, output_dir, active_years, args.mode)
    after = new_numeric_snapshot(yearly, combined)
    assert_numeric_unchanged(before, after)
    write_summaries(output_dir, yearly, combined)
    manifest = build_manifest(
        records=records,
        output_dir=output_dir,
        active_years=active_years,
        model_seeds=MODEL_SEEDS,
        attack_seed=ATTACK_SEED,
        mode=args.mode,
        root=ROOT,
        here=HERE,
        control=CONTROL,
        notice=NOTICE,
        builder_path=Path(__file__),
        previous_manifest=previous_manifest,
    )
    write_json_atomic(output_dir / "manifest.json", manifest)
    report = {
        "status": "rebuilt_without_attack_rerun",
        "output_dir": str(output_dir),
        "records_read": len(records),
        "record_keys_exact": inventory["all_expected_records"],
        "numeric_snapshot_sha256_before": snapshot_digest(before),
        "numeric_snapshot_sha256_after": snapshot_digest(after),
        "numeric_values_unchanged": snapshot_digest(before) == snapshot_digest(after),
        "statistical_recoverability_records": manifest["statistical_recoverability_records"],
        "practical_leakage_records": manifest["practical_leakage_records"],
        "models_hashed": len(manifest["model_sha256"]),
        "data_pack_files_hashed": len(manifest["data_pack_sha256"]),
    }
    write_json_atomic(output_dir / "rebuild_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
