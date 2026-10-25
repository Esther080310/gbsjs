"""Small, label-free smoke test of FiLM-L message extraction and wire frames."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from protocol import (BLOCKS, HEADER, deserialize_block, extract_messages, fit_scales,
                      fusion_from_messages, load_model, quantize_dequantize,
                      serialize_block)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def main():
    torch.set_num_threads(4)
    year, seed = 2020, 42
    formal, pack, model = load_model(year, seed)
    train = pack["splits"][0]
    test = pack["splits"][3]
    fit_messages = extract_messages(model, train, slice(0, 512))
    messages = extract_messages(model, test, slice(0, 64))
    scales = fit_scales(fit_messages)

    expected_shapes = {
        "market": (64, 17),
        "finance": (64, 89),
        "governance": (64, 22),
        "audit_control": (64, 9),
    }
    assert {k: v.shape for k, v in messages.items()} == expected_shapes
    assert sum(x.shape[1] for x in messages.values()) == 137
    assert set(np.unique(messages["audit_control"][:, 8])).issubset({0.0, 1.0})

    # The four-message decomposition must reproduce the original unquantized model exactly.
    with torch.no_grad():
        xs = [torch.as_tensor(x[:64], dtype=torch.float32) for x in test["xs"]]
        direct_a, direct_b = model(
            xs, torch.as_tensor(test["industry"][:64]),
            torch.as_tensor(test["q"][:64]), torch.as_tensor(test["state"][:64])
        )
    wire_a, wire_b = fusion_from_messages(model, test["industry"][:64], messages)
    np.testing.assert_array_equal(direct_a.numpy(), wire_a)
    np.testing.assert_array_equal(direct_b.numpy(), wire_b)

    rounds = {}
    for precision in ("fp32", "fp16", "int8"):
        decoded = {}
        frame_bytes = 0
        max_message_error = 0.0
        for block in BLOCKS:
            scale = scales[block] if precision == "int8" else None
            frame = serialize_block(block, messages[block], precision, scale)
            b2, p2, values = deserialize_block(frame, scale)
            assert b2 == block and p2 == precision
            expected = quantize_dequantize(block, messages[block], precision, scale)
            np.testing.assert_array_equal(values, expected)
            decoded[block] = values
            frame_bytes += len(frame)
            max_message_error = max(max_message_error,
                                    float(np.max(np.abs(values - messages[block]))))
        a, b = fusion_from_messages(model, test["industry"][:64], decoded)
        payload_bytes = frame_bytes - 4 * HEADER.size
        rounds[precision] = {
            "rows": 64,
            "four_frames_bytes": frame_bytes,
            "header_bytes": 4 * HEADER.size,
            "payload_bytes_per_row": payload_bytes / 64,
            "one_company_four_frame_bytes": payload_bytes / 64 + 4 * HEADER.size,
            "bytes_per_row_including_amortized_headers": frame_bytes / 64,
            "max_message_abs_error": max_message_error,
            "max_overall_logit_abs_error": float(np.max(np.abs(a - wire_a))),
            "max_type_logit_abs_error": float(np.max(np.abs(b - wire_b))),
            "message_sha256": {k: digest(v) for k, v in decoded.items()},
        }

    # Seed-specific messages must not silently be treated as a common ensemble message.
    # This is a topology check only: it uses raw model logits and no labels.
    seeds = (42, 100, 17, 223, 483, 907)
    models, seed_messages, raw_outputs = [model], [messages], [(wire_a, wire_b)]
    for other_seed in seeds[1:]:
        _, other_pack, other_model = load_model(year, other_seed)
        other_messages = extract_messages(other_model, other_pack["splits"][3], slice(0, 64))
        other_a, other_b = fusion_from_messages(
            other_model, other_pack["splits"][3]["industry"][:64], other_messages
        )
        models.append(other_model)
        seed_messages.append(other_messages)
        raw_outputs.append((other_a, other_b))

    # q1/q2/q3 and r are input metadata and are exactly common across seeds.
    for block in BLOCKS:
        for other in seed_messages[1:]:
            np.testing.assert_array_equal(messages[block][:, -1], other[block][:, -1])
    seed_difference = {
        k: max(float(np.max(np.abs(messages[k][:, :-1] - other[k][:, :-1])))
               for other in seed_messages[1:])
        for k in BLOCKS
    }
    assert all(v > 0 for v in seed_difference.values())

    averaged_messages = {
        block: np.mean([m[block] for m in seed_messages], axis=0).astype(np.float32)
        for block in BLOCKS
    }
    # This remains a label-free, uncalibrated topology check. Compare mean raw
    # logits so its algebra matches the frozen formal predictor. The formal
    # evaluator applies each seed's calibration before averaging.
    target_a = np.mean([a for a, _ in raw_outputs], axis=0)
    target_b = np.mean([b for _, b in raw_outputs], axis=0)
    averaged_message_outputs = [
        fusion_from_messages(m, test["industry"][:64], averaged_messages) for m in models
    ]
    naive_a = np.mean([a for a, _ in averaged_message_outputs], axis=0)
    naive_b = np.mean([b for _, b in averaged_message_outputs], axis=0)
    naive_error = {
        "overall_logit_max_abs_error_raw_uncalibrated_smoke": float(np.max(np.abs(target_a - naive_a))),
        "type_logit_max_abs_error_raw_uncalibrated_smoke": float(np.max(np.abs(target_b - naive_b))),
    }
    assert naive_error["overall_logit_max_abs_error_raw_uncalibrated_smoke"] > 0
    assert naive_error["type_logit_max_abs_error_raw_uncalibrated_smoke"] > 0

    out = {
        "status": "passed",
        "scope": "2020 seed42, 64 rows; shapes/round-trip only, no labels and no policy selection",
        "physical_topology": "three data nodes, one transport round; X3 carries governance and audit-control logical blocks",
        "logical_shapes": {k: list(v) for k, v in expected_shapes.items()},
        "unquantized_manual_forward_exact": True,
        "roundtrips": rounds,
        "seed42_vs_other_seeds_max_content_difference": seed_difference,
        "six_seed_messages_required": True,
        "naive_average_message_is_not_exact": naive_error,
        "ensemble_rule_for_formal_evaluation": "fuse each seed's messages to raw logits; apply that seed's calibration; equal-average six calibrated logits; apply sigmoid once",
        "int8_scales_fitted_from": "first 512 training rows; smoke only, not formal deployment scales",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "preflight.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

