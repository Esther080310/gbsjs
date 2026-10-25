"""Select the smallest deployment payload among candidates passing the frozen strict gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    three_path = RESULTS / "formal_candidate_three_year.csv"
    year_path = RESULTS / "formal_candidate_year.csv"
    byte_path = RESULTS / "formal_candidate_bytes.csv"
    output = RESULTS / "strict_candidate_selection.json"
    if output.exists():
        raise FileExistsError(output)
    three = pd.read_csv(three_path)
    yearly = pd.read_csv(year_path)
    byte = pd.read_csv(byte_path)
    merged = three.merge(byte, on=["candidate_number", "candidate_id"], validate="1:1")
    assert len(merged) == 81
    passed = merged[merged["strict.all_required_pass_all_years"]].copy()
    assert len(passed) > 0
    # The final predictor is the six-model ensemble.  Payload is therefore the
    # primary size criterion; candidate number is a deterministic tie breaker.
    chosen = passed.sort_values([
        "six_model_shared_q_r.payload_bytes_per_company",
        "three_year_inference_plus_scale_bytes",
        "candidate_number",
    ], kind="stable").iloc[0]
    rows = yearly[yearly.candidate_number == chosen.candidate_number].sort_values("year")
    assert len(rows) == 3 and rows["strict.all_required_pass"].all()

    fp32 = merged.loc[merged.candidate_number == 1].iloc[0]
    all_int8 = merged.loc[merged.candidate_number == 81].iloc[0]
    baseline_blocks = {
        "x1_market": int(fp32["six_model_shared_q_r.payload_bytes_by_block.market"]),
        "x2_finance": int(fp32["six_model_shared_q_r.payload_bytes_by_block.finance"]),
        "x3_governance_plus_audit": int(
            fp32["six_model_shared_q_r.payload_bytes_by_block.governance"] +
            fp32["six_model_shared_q_r.payload_bytes_by_block.audit_control"]),
    }
    chosen_blocks = {
        "x1_market": int(chosen["six_model_shared_q_r.payload_bytes_by_block.market"]),
        "x2_finance": int(chosen["six_model_shared_q_r.payload_bytes_by_block.finance"]),
        "x3_governance_plus_audit": int(
            chosen["six_model_shared_q_r.payload_bytes_by_block.governance"] +
            chosen["six_model_shared_q_r.payload_bytes_by_block.audit_control"]),
    }
    node_reduction = {key: 1.0 - chosen_blocks[key] / baseline_blocks[key]
                      for key in baseline_blocks}

    # Analytic wire-time scenarios for a 64-company batch.  These are not a
    # measured network benchmark; one RTT is added because the protocol is one round.
    def scenario(row, mbps: float, rtt_ms: float, shared: bool) -> float:
        if shared:
            body = float(row["six_model_shared_q_r.payload_bytes_per_company"]) * 64 + 80
        else:
            x1 = float(row["six_model_shared_q_r.payload_bytes_by_block.market"]) * 64 + 20
            x2 = float(row["six_model_shared_q_r.payload_bytes_by_block.finance"]) * 64 + 20
            x3 = (float(row["six_model_shared_q_r.payload_bytes_by_block.governance"]) +
                  float(row["six_model_shared_q_r.payload_bytes_by_block.audit_control"])) * 64 + 40
            body = max(x1, x2, x3)
        return rtt_ms + body * 8 / (mbps * 1_000_000) * 1000

    latency = []
    for topology in ("shared_uplink", "independent_node_links"):
        for mbps in (1, 10, 100):
            for rtt in (1, 10, 50):
                base_ms = scenario(fp32, mbps, rtt, topology == "shared_uplink")
                chosen_ms = scenario(chosen, mbps, rtt, topology == "shared_uplink")
                latency.append({
                    "topology": topology, "bandwidth_mbps": mbps, "rtt_ms": rtt,
                    "fp32_ms": base_ms, "chosen_ms": chosen_ms,
                    "absolute_saving_ms": base_ms - chosen_ms,
                    "end_to_end_fraction": chosen_ms / base_ms,
                })

    result = {
        "status": "strict-gate minimum-payload candidate selected",
        "selection_rule": (
            "among candidates passing every frozen strict condition in every year, "
            "minimize final six-model payload bytes per company"
        ),
        "candidates_total": 81,
        "strict_pass_count": int(len(passed)),
        "secondary_pass_count": int(merged[
            "secondary.all_secondary_thresholds_pass_all_years"].sum()),
        "selected": {
            "candidate_number": int(chosen.candidate_number),
            "candidate_id": chosen.candidate_id,
            "precision": {
                "x1_market": "fp16", "x2_finance": "int8",
                "x3_governance": "int8", "x3_audit_control": "int8",
            },
            "single_model_payload_bytes": int(chosen[
                "single_model.payload_bytes_per_company"]),
            "six_model_payload_bytes": int(chosen[
                "six_model_shared_q_r.payload_bytes_per_company"]),
            "six_model_fp32_payload_bytes": int(fp32[
                "six_model_shared_q_r.payload_bytes_per_company"]),
            "six_model_payload_reduction": float(1.0 -
                chosen["six_model_shared_q_r.payload_bytes_per_company"] /
                fp32["six_model_shared_q_r.payload_bytes_per_company"]),
            "three_year_inference_plus_scale_bytes": int(chosen[
                "three_year_inference_plus_scale_bytes"]),
            "three_year_fp32_inference_bytes": int(fp32[
                "three_year_inference_plus_scale_bytes"]),
            "node_payload_bytes_fp32": baseline_blocks,
            "node_payload_bytes_selected": chosen_blocks,
            "node_payload_reduction": node_reduction,
            "maximum_overall_probability_error": float(chosen[
                "strict.max_overall_probability_absolute_error"]),
            "maximum_type_probability_error": float(chosen[
                "strict.max_type_probability_absolute_error"]),
            "yearly": rows.to_dict("records"),
        },
        "all_int8_sensitivity": {
            "candidate_number": 81,
            "six_model_payload_bytes": int(all_int8[
                "six_model_shared_q_r.payload_bytes_per_company"]),
            "payload_reduction": float(1.0 -
                all_int8["six_model_shared_q_r.payload_bytes_per_company"] /
                fp32["six_model_shared_q_r.payload_bytes_per_company"]),
            "strict_pass": bool(all_int8["strict.all_required_pass_all_years"]),
            "secondary_pass": bool(all_int8[
                "secondary.all_secondary_thresholds_pass_all_years"]),
            "reason_not_primary": (
                "2021 changed one company in/out of the Top10% set and one selected "
                "company's Top2 type set; it therefore fails the frozen exact-set gate"
            ),
        },
        "analytic_latency_scenarios_batch64": latency,
        "claim_boundary": (
            "communication encoding for inference only; no privacy guarantee and no "
            "measured TCP/TLS or physical multi-machine latency"
        ),
        "source_sha256": {str(path): digest(path)
                          for path in (three_path, year_path, byte_path,
                                       HERE / "protocol_contract.json",
                                       HERE / "privacy_interface_contract.json")},
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    pd.DataFrame(latency).to_csv(RESULTS / "analytic_latency_scenarios_batch64.csv",
                                 index=False)
    print(json.dumps({"selected": result["selected"],
                      "strict_pass_count": result["strict_pass_count"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
