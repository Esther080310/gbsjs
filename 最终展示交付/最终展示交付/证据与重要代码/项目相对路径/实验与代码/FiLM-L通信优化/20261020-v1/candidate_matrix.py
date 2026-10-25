"""Generate the frozen 3^4 communication candidate matrix.

This program performs accounting only. It never opens labels, predictions or
test metrics and therefore cannot select a precision policy.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BLOCKS = ("market", "finance", "governance", "audit_control")
PRECISIONS = ("fp32", "fp16", "int8")
ITEM_BYTES = {"fp32": 4, "fp16": 2, "int8": 1}
HEADER_BYTES = 20
FRAMES = 4

# Single-model continuous values. audit_control's exact binary r is separate.
SINGLE_DIMS = {"market": 17, "finance": 89, "governance": 22, "audit_control": 8}
# Six seed-specific encodings/control codes; q1/q2/q3 occur once and r is separate.
ENSEMBLE_SHARED_DIMS = {
    "market": 6 * 16 + 1,
    "finance": 6 * 88 + 1,
    "governance": 6 * 21 + 1,
    "audit_control": 6 * 8,
}


def account(policy: dict[str, str], dims: dict[str, int]) -> dict:
    by_block = {
        block: dims[block] * ITEM_BYTES[policy[block]] + (1 if block == "audit_control" else 0)
        for block in BLOCKS
    }
    payload = sum(by_block.values())
    scales = 4 * sum(dims[b] for b in BLOCKS if policy[b] == "int8")
    return {
        "payload_bytes_per_company": payload,
        "payload_bytes_by_block": by_block,
        "one_company_four_frame_bytes": payload + FRAMES * HEADER_BYTES,
        "batch64_bytes_per_company": payload + FRAMES * HEADER_BYTES / 64,
        "one_time_int8_scale_manifest_bytes": scales,
    }


def main() -> None:
    rows = []
    for index, choices in enumerate(itertools.product(PRECISIONS, repeat=len(BLOCKS)), start=1):
        policy = dict(zip(BLOCKS, choices))
        rows.append({
            "candidate_number": index,
            "candidate_id": "__".join(f"{b}-{policy[b]}" for b in BLOCKS),
            "precision_by_logical_block": policy,
            "single_model": account(policy, SINGLE_DIMS),
            "six_model_logit_ensemble_shared_q_r": account(policy, ENSEMBLE_SHARED_DIMS),
        })
    assert len(rows) == 81 and len({r["candidate_id"] for r in rows}) == 81
    baseline_single = rows[0]["single_model"]["payload_bytes_per_company"]
    baseline_ensemble = rows[0]["six_model_logit_ensemble_shared_q_r"]["payload_bytes_per_company"]
    assert baseline_single == 545 and baseline_ensemble == 3205
    for row in rows:
        row["single_model"]["payload_fraction_of_fp32"] = (
            row["single_model"]["payload_bytes_per_company"] / baseline_single
        )
        row["six_model_logit_ensemble_shared_q_r"]["payload_fraction_of_fp32"] = (
            row["six_model_logit_ensemble_shared_q_r"]["payload_bytes_per_company"] / baseline_ensemble
        )
    out = {
        "status": "candidate accounting only; formal evaluation not run",
        "count": len(rows),
        "block_order": list(BLOCKS),
        "precision_order": list(PRECISIONS),
        "ensemble_rule": "equal-average six calibrated logits, then apply sigmoid",
        "physical_topology": {
            "data_nodes": 3,
            "uplinks": ["x1_market", "x2_finance", "x3_governance_and_audit"],
            "logical_frames_per_batch": FRAMES,
            "transport_rounds": 1,
        },
        "serialization": {
            "header_bytes_per_logical_frame": HEADER_BYTES,
            "r": "one exact uint8 per company",
            "int8_scale": "FP32 per channel, pre-shared once with model manifest",
            "requests_tls_encryption": "excluded",
        },
        "candidates": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / "candidate_manifest.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({
        "status": out["status"],
        "candidates": len(rows),
        "output": str(target),
        "single_fp32_payload_bytes": baseline_single,
        "ensemble_fp32_payload_bytes": baseline_ensemble,
        "single_all_int8_payload_bytes": rows[-1]["single_model"]["payload_bytes_per_company"],
        "ensemble_all_int8_payload_bytes": rows[-1]["six_model_logit_ensemble_shared_q_r"]["payload_bytes_per_company"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
