"""FiLM-L privacy-audit preflight.

This script never opens the formal target split.  It checks message extraction on the first
384 rows of the 2020 attack-training split and runs one shape-only Ridge smoke test.
No smoke metric is used for attack selection.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTROL = ROOT / "实验与代码/控制消息架构/20261018-v1"
NOTICE = ROOT / "实验与代码/违规通报结构化研究"
sys.path[:0] = [str(CONTROL), str(NOTICE), str(NOTICE / "scripts")]

import cma_lib as C  # noqa: E402
import ablation_lib as A  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def extract(model: A.FilmKeepX3, split: dict, n: int) -> dict[str, np.ndarray]:
    model.eval()
    xs = [torch.as_tensor(x[:n]) for x in split["xs"]]
    q = torch.as_tensor(split["q"][:n])
    state = torch.as_tensor(split["state"][:n])
    industry = torch.as_tensor(split["industry"][:n])

    h1 = model.base.enc[0](xs[0])
    h2 = model.base.enc[1](xs[1])
    h3 = model.base.enc[2](xs[2], state)
    a = xs[2][:, model.old_x3_dimension:model.old_x3_dimension + C.AUDIT_DIM]
    r = a[:, :1]
    c = model.control(a)
    gamma, beta = model.film(c).chunk(2, dim=1)
    h2_post = h2 + r * (gamma * h2 + beta)

    m1 = torch.cat([h1, q[:, 0:1]], dim=1)
    m2 = torch.cat([h2_post, q[:, 1:2]], dim=1)
    m3 = torch.cat([h3, q[:, 2:3]], dim=1)

    # Run the canonical forward and prove our implementation transcript is exact.
    _ = model(xs, industry, q, state)
    for expected, actual in zip((m1, m2, m3), model.messages):
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)

    deployment = torch.cat([h1, h2, h3, q, c, r, industry[:, None].float()], dim=1)
    implementation = torch.cat([m1, m2, m3, industry[:, None].float()], dim=1)
    return {
        "h1": h1.numpy(), "h2": h2.numpy(), "h2_post": h2_post.numpy(),
        "h3": h3.numpy(), "c": c.numpy(), "r": r.numpy(), "q": q.numpy(),
        "deployment_view": deployment.numpy(), "implementation_view": implementation.numpy(),
    }


def main() -> None:
    torch.set_num_threads(2)
    seed, year, n = 42, 2020, 384
    # Deliberately request only fold 0. No formal target labels or predictions are opened.
    pack0 = C.pack_for(year, range(1))
    # Model construction only needs fold-0 dimensions/names. Reuse this object so the
    # preflight does not even materialise the attack-selection or formal-evaluation folds.
    architecture_pack = pack0
    formal = C.shared.load_loader().formal
    model = A.FilmKeepX3(formal, architecture_pack, seed)
    model_path = CONTROL / f"artifacts/{year}/FiLM-L_s{seed}/model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    split = pack0["splits"][0]
    assert len(split["y"]) >= n
    z = extract(model, split, n)

    expected = {
        "h1": (n, 16), "h2": (n, 88), "h2_post": (n, 88), "h3": (n, 21),
        "c": (n, 8), "r": (n, 1), "q": (n, 3),
        # deployment includes h1+h2+h3+q+c+r+industry; implementation includes 128 floats+industry
        "deployment_view": (n, 138), "implementation_view": (n, 129),
    }
    for key, shape in expected.items():
        assert z[key].shape == shape, (key, z[key].shape, shape)
        assert np.isfinite(z[key]).all(), key

    # Shape/finite smoke only. Performance is intentionally not calculated or saved.
    attack_train, attack_smoke = np.arange(256), np.arange(256, 384)
    ridge = Ridge(alpha=1.0).fit(z["h1"][attack_train], split["xs"][0][attack_train])
    prediction = ridge.predict(z["h1"][attack_smoke])
    assert prediction.shape == (128, split["xs"][0].shape[1])
    assert np.isfinite(prediction).all()

    # Six-model joint deployment surface. q/r/industry are shared once; every seed-specific
    # h1/h2/h3/c is retained. This is a shape/information-surface preflight, not an attack.
    seed_views = []
    deployment_seed_blocks = []
    implementation_seed_blocks = []
    for ensemble_seed in C.SEEDS:
        ensemble_model = A.FilmKeepX3(formal, architecture_pack, ensemble_seed)
        ensemble_path = CONTROL / f"artifacts/{year}/FiLM-L_s{ensemble_seed}/model.pt"
        ensemble_ckpt = torch.load(ensemble_path, map_location="cpu", weights_only=True)
        ensemble_model.load_state_dict(ensemble_ckpt["state_dict"])
        ev = extract(ensemble_model, split, 64)
        seed_views.append(ev)
        deployment_seed_blocks.append(
            np.concatenate([ev["h1"], ev["h2"], ev["h3"], ev["c"]], axis=1))
        implementation_seed_blocks.append(
            np.concatenate([ev["h1"], ev["h2_post"], ev["h3"]], axis=1))
    public_industry = split["industry"][:64, None].astype(np.float32)
    deployment_shared = np.concatenate(
        [seed_views[0]["q"], seed_views[0]["r"], public_industry], axis=1)
    implementation_shared = np.concatenate(
        [seed_views[0]["q"], public_industry], axis=1)
    deployment_ensemble_view = np.concatenate(
        [*deployment_seed_blocks, deployment_shared], axis=1)
    implementation_ensemble_view = np.concatenate(
        [*implementation_seed_blocks, implementation_shared], axis=1)
    assert deployment_ensemble_view.shape == (64, 803)  # 6*(16+88+21+8)+q3+r1+industry1
    assert implementation_ensemble_view.shape == (64, 754)  # 6*(16+88+21)+q3+industry1
    assert np.isfinite(deployment_ensemble_view).all()
    assert np.isfinite(implementation_ensemble_view).all()
    pair_differences = [float(np.max(np.abs(deployment_seed_blocks[i] - deployment_seed_blocks[0])))
                        for i in range(1, len(deployment_seed_blocks))]
    assert all(v > 0 for v in pair_differences)

    report = {
        "status": "preflight_only_no_formal_attack_metrics",
        "formal_target_split_opened": False,
        "year": year,
        "seed": seed,
        "rows_used_from_attack_training_split": n,
        "source_input_shapes": [list(x.shape) for x in split["xs"]],
        "message_shapes": {k: list(v.shape) for k, v in z.items()},
        "six_model_deployment_joint_view_shape": list(deployment_ensemble_view.shape),
        "six_model_implementation_joint_view_shape": list(implementation_ensemble_view.shape),
        "six_model_seed_blocks_nonidentical": True,
        "six_model_max_abs_difference_vs_seed42": pair_differences,
        "canonical_forward_messages_exact": True,
        "ridge_shape_smoke_passed": True,
        "implementation_payload_floats_excluding_industry": 128,
        "declared_deployment_payload_floats_excluding_industry": 137,
        "declared_deployment_protocol_floats_with_audit_control_and_q": 137,
        "report_accounting_floats_if_industry_excluded_as_public_context": 137,
        "six_model_deployment_joint_attack_features_excluding_public_industry": 802,
        "six_model_implementation_joint_attack_features_excluding_public_industry": 753,
        "six_model_deployment_physical_payload_floats_without_deduplication": 822,
        "six_model_implementation_physical_payload_floats_without_deduplication": 768,
        "note": (
            "The report's 137-float accounting counts 128 main-message floats plus c(8)+r(1). "
            "The explicit deployment tensor here contains h1(16)+h2(88)+h3(21)+q(3)+c(8)+r(1)=137; "
            "industry is appended only for attack-view construction and is excluded from payload accounting."
        ),
        "source_sha256": {
            str(CONTROL / "ablation_lib.py"): sha256(CONTROL / "ablation_lib.py"),
            str(CONTROL / "cma_lib.py"): sha256(CONTROL / "cma_lib.py"),
            str(model_path): sha256(model_path),
        },
    }
    assert z["deployment_view"].shape[1] - 1 == 137  # last column is public industry context
    out = HERE / "results/preflight.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

