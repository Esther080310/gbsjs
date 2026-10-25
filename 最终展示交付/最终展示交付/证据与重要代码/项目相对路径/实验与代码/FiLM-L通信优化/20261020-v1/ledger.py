"""Build a protocol accounting ledger without evaluating labels."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTROL = ROOT / "实验与代码/控制消息架构/20261018-v1"
RESULTS = HERE / "results"
SEEDS = (42, 100, 17, 223, 483, 907)
YEARS = (2020, 2021, 2022)


def main():
    # Model-slot accounting retained for compatibility with the midterm report.
    single = {
        "market": 17,
        "finance": 89,
        "governance": 22,
        "audit_control": 9,
    }
    single_total = sum(single.values())
    assert single_total == 137

    # Six models have seed-specific h1/h2/h3/c. q1/q2/q3 and r are common metadata.
    seed_specific = 16 + 88 + 21 + 8
    common = 3 + 1
    ensemble_shared = len(SEEDS) * seed_specific + common
    assert ensemble_shared == 802

    training = []
    for year in YEARS:
        for seed in SEEDS:
            p = CONTROL / "artifacts" / str(year) / f"FiLM-L_s{seed}" / "run.json"
            d = json.loads(p.read_text(encoding="utf8"))
            updates_f = d["train_forward_bytes"] / (4 * 128)
            updates_b = d["train_backward_bytes"] / (4 * 125)
            val_rows = d["epoch_validation_bytes"] / (4 * 128)
            assert updates_f == int(updates_f) and updates_b == int(updates_b)
            assert int(updates_f) == int(updates_b)
            training.append({
                "year": year,
                "seed": seed,
                "epochs": d["epochs"],
                "sample_updates": int(updates_f),
                "recorded_forward_bytes_128": d["train_forward_bytes"],
                "corrected_forward_bytes_137": int(updates_f * 137 * 4),
                "recorded_backward_bytes_125": d["train_backward_bytes"],
                "corrected_backward_bytes_133": int(updates_b * 133 * 4),
                "recorded_validation_bytes_128": d["epoch_validation_bytes"],
                "corrected_validation_bytes_137": int(val_rows * 137 * 4),
            })

    payload_by_precision = {
        # r is represented as one exact byte on the wire.  This 3-byte difference from
        # the legacy 548-byte model-slot account is packing, not a learned saving.
        "all_fp32": 17 * 4 + 89 * 4 + 22 * 4 + 8 * 4 + 1,
        "all_fp16": 17 * 2 + 89 * 2 + 22 * 2 + 8 * 2 + 1,
        "all_int8": 17 + 89 + 22 + 8 + 1,
    }
    header_bytes = 20
    logical_frames = 4
    single_by_block = {
        "market": {"float_slots": 17, "fp32_payload_bytes": 68},
        "finance": {"float_slots": 89, "fp32_payload_bytes": 356},
        "governance": {"float_slots": 22, "fp32_payload_bytes": 88},
        "audit_control": {
            "float_slots": 9,
            "legacy_fp32_bytes": 36,
            "packed_fp32_payload_bytes": 33,
            "note": "c为8个FP32，r为1个uint8",
        },
    }
    ensemble_by_block = {
        # Six seed-specific encodings/control codes, one shared q per node and
        # one shared r.  Model order and dimensions live in a pre-shared manifest.
        "market": {"continuous_values": 6 * 16 + 1, "fp32_payload_bytes": (6 * 16 + 1) * 4},
        "finance": {"continuous_values": 6 * 88 + 1, "fp32_payload_bytes": (6 * 88 + 1) * 4},
        "governance": {"continuous_values": 6 * 21 + 1, "fp32_payload_bytes": (6 * 21 + 1) * 4},
        "audit_control": {"continuous_values": 6 * 8, "packed_fp32_payload_bytes": 6 * 8 * 4 + 1},
    }
    ensemble_packed_fp32 = sum(
        x.get("fp32_payload_bytes", x.get("packed_fp32_payload_bytes", 0))
        for x in ensemble_by_block.values()
    )
    assert ensemble_packed_fp32 == 3205
    recorded = int(sum(x["recorded_forward_bytes_128"] + x["recorded_backward_bytes_125"] + x["recorded_validation_bytes_128"] for x in training))
    corrected = int(sum(x["corrected_forward_bytes_137"] + x["corrected_backward_bytes_133"] + x["corrected_validation_bytes_137"] for x in training))
    ledger = {
        "scope": "preflight accounting; no label evaluation and no 81-arm selection",
        "topology": {
            "physical_data_nodes": 3,
            "physical_uplinks": ["x1_market", "x2_finance", "x3_governance_and_audit"],
            "logical_blocks": ["market", "finance", "governance", "audit_control"],
            "transport_rounds": 1,
            "note": "X3同一连接承载治理表示和审计控制支路；若未来审计由独立机构持有，物理上行增至4，但数值payload不变",
        },
        "serialization_assumptions": {
            "frame_header_bytes_per_logical_block": header_bytes,
            "logical_frames_per_batched_single_model_request": logical_frames,
            "header_fields": "magic/version/block/precision/flags/row_count/dimension/body_length",
            "int8_scales": "per-model, per-logical-block, per-channel; pre-shared deployment metadata",
            "model_order_and_dimensions": "pre-shared deployment manifest",
            "requests_tls_encryption": "not included",
        },
        "single_model": {
            "float_slots_by_block": single,
            "fp32_account_by_block": single_by_block,
            "float_slots": single_total,
            "legacy_numeric_bytes_fp32": single_total * 4,
            "wire_payload_bytes_with_r_packed": payload_by_precision,
            "one_company_four_frame_bytes": {k: v + logical_frames * header_bytes for k, v in payload_by_precision.items()},
            "batch64_bytes_per_company": {k: v + logical_frames * header_bytes / 64 for k, v in payload_by_precision.items()},
            "int8_scale_manifest_bytes_if_all_int8": (17 + 89 + 22 + 8) * 4,
        },
        "six_model_ensemble": {
            "aggregation": "six calibrated logits are equally averaged, then sigmoid is applied",
            "naive_repeated_float_slots": single_total * len(SEEDS),
            "naive_repeated_fp32_bytes": single_total * len(SEEDS) * 4,
            "naive_repeated_packed_fp32_payload_bytes": payload_by_precision["all_fp32"] * len(SEEDS),
            "naive_repeated_one_company_24_frame_bytes": payload_by_precision["all_fp32"] * len(SEEDS) + logical_frames * len(SEEDS) * header_bytes,
            "naive_repeated_batch64_bytes_per_company": payload_by_precision["all_fp32"] * len(SEEDS) + logical_frames * len(SEEDS) * header_bytes / 64,
            "shared_q_and_r_float_slots": ensemble_shared,
            "shared_q_and_r_fp32_equivalent_bytes": ensemble_shared * 4,
            "packed_by_logical_block_fp32_payload_bytes": ensemble_packed_fp32,
            "packed_by_logical_block_one_company_frame_bytes": ensemble_packed_fp32 + logical_frames * header_bytes,
            "packed_by_logical_block_batch64_bytes_per_company": ensemble_packed_fp32 + logical_frames * header_bytes / 64,
            "packed_fp32_by_block": ensemble_by_block,
            "all_fp16_packed_payload_bytes": (6 * 16 + 1) * 2 + (6 * 88 + 1) * 2 + (6 * 21 + 1) * 2 + 6 * 8 * 2 + 1,
            "all_int8_packed_payload_bytes": (6 * 16 + 1) + (6 * 88 + 1) + (6 * 21 + 1) + 6 * 8 + 1,
            "int8_scale_manifest_bytes_if_all_int8": ((6 * 16 + 1) + (6 * 88 + 1) + (6 * 21 + 1) + 6 * 8) * 4,
            "exact_local_average_possible": False,
            "reason": "encoders, control maps and nonlinear fusion heads differ by seed; one averaged message does not reproduce the mean of six per-seed calibrated logits",
        },
        "training_accounting": {
            "kind": "numeric-slot compatibility estimate, not measured network traffic",
            "old_forward_slots": 128,
            "corrected_forward_slots": 137,
            "old_backward_differentiable_slots": 125,
            "corrected_backward_differentiable_slots": 133,
            "runs": training,
            "aggregate_recorded_bytes": recorded,
            "aggregate_corrected_bytes": corrected,
            "correction_percent": 100 * (corrected / recorded - 1),
        },
        "not_counted_yet": [
            "requests and acknowledgements",
            "TCP/TLS or encryption expansion",
            "model-weight deployment (INT8 scale manifest is separately shown above)",
            "industry code if it is not already held by the fusion server",
            "wall-clock network latency and packet loss",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "protocol_ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps({
        "single_fp32_bytes": single_total * 4,
        "ensemble_naive_fp32_bytes": single_total * len(SEEDS) * 4,
        "ensemble_shared_metadata_fp32_bytes": ensemble_shared * 4,
        "training_recorded_GB": ledger["training_accounting"]["aggregate_recorded_bytes"] / 1e9,
        "training_corrected_GB": ledger["training_accounting"]["aggregate_corrected_bytes"] / 1e9,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

