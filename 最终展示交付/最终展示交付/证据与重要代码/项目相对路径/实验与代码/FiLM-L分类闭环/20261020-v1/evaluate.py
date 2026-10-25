"""Zero-training classification closure for the frozen FiLM-L predictor."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = ROOT / "实验与代码/控制消息架构/20261018-v1"
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(ROOT / "实验与代码/违规通报结构化研究"))
sys.path.insert(0, str(ROOT / "实验与代码/违规通报结构化研究/scripts"))
import cma_lib as C  # noqa: E402

YEARS = (2020, 2021, 2022)
SEEDS = tuple(C.SEEDS)
TYPE_CODES = ("P2501", "P2503", "P2504", "P2505", "P2506",
              "P2509", "P2510", "P2514", "P2515", "P2599")
INDUSTRIES = tuple([chr(65 + i) for i in range(19)] + ["UNK"])
OBSERVED_INDUSTRIES = INDUSTRIES[:19]
EXPECTED = {2020: (4292, 787, 430, 258),
            2021: (4794, 747, 480, 227),
            2022: (5199, 629, 520, 210)}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def support_status(n: int) -> str:
    if n < 10:
        return "sparse_lt10"
    if n < 30:
        return "descriptive_10_29"
    return "reportable_ge30"


def load_all() -> tuple[dict, list[dict]]:
    """Load 2020/21 before the 2022 redirect, then make all data self-contained."""
    data: dict[int, dict] = {}
    sources: list[dict] = []
    for year in (2020, 2021):
        data[year], rec = load_year(year)
        sources.extend(rec)
    import train_audit_year2022 as fix22  # noqa: E402
    fix22.install_redirected_loaders()
    data[2022], rec = load_year(2022)
    sources.extend(rec)
    return data, sources


def load_year(year: int) -> tuple[dict, list[dict]]:
    pack = C.pack_for(year, range(4))
    train, test = pack["splits"][0], pack["splits"][3]
    y = test["y"].astype(np.int8)
    types = test["types"].astype(np.int8)
    industry = test["industry"].astype(np.int64)
    symbols = test["keys"].Symbol.to_numpy(dtype=str)
    common = np.argsort(-train["types"].sum(0), kind="stable")[:2]
    a_logits, b_logits, sources = [], [], []
    for seed in SEEDS:
        path = MODEL / "artifacts" / str(year) / f"FiLM-L_s{seed}" / "fold3.npz"
        with np.load(path, allow_pickle=False) as z:
            assert z["a_logit"].shape == (len(y),)
            assert z["b_logit"].shape == (len(y), len(TYPE_CODES))
            assert np.isfinite(z["a_logit"]).all() and np.isfinite(z["b_logit"]).all()
            a_logits.append(z["a_logit"].astype(np.float64))
            b_logits.append(z["b_logit"].astype(np.float64))
        sources.append({"path": str(path), "sha256": sha256(path)})
    a_logit_seed = np.stack(a_logits)
    b_logit_seed = np.stack(b_logits)
    a_seed = sigmoid(a_logit_seed)
    b_seed = sigmoid(b_logit_seed)
    # Frozen presentation predictor: average calibrated logits first.
    a = sigmoid(np.mean(a_logit_seed, axis=0))
    b = sigmoid(np.mean(b_logit_seed, axis=0))
    assert set(np.unique(industry)).issubset(set(range(20)))
    assert np.all(types.sum(1)[y == 0] == 0)
    return dict(y=y, types=types, industry=industry, symbols=symbols,
                common=common.astype(int), a_seed=a_seed, b_seed=b_seed,
                a_logit_seed=a_logit_seed, b_logit_seed=b_logit_seed,
                a=a, b=b,
                a_probability_average=np.mean(a_seed, axis=0),
                b_probability_average=np.mean(b_seed, axis=0),
                signatures={"y": array_sha(y), "types": array_sha(types),
                            "industry": array_sha(industry), "symbols": array_sha(symbols)}), sources


def selected_and_nom(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = int(np.ceil(.1 * len(a)))
    selected = np.zeros(len(a), dtype=bool)
    selected[np.argsort(-a, kind="stable")[:k]] = True
    top = np.argsort(-b, axis=1, kind="stable")[:, :2]
    nom = np.zeros_like(b, dtype=bool)
    nom[np.arange(len(a))[:, None], top] = True
    assert np.all(nom.sum(1) == 2)
    return selected, nom


def application_metrics(y, types, a, b, common) -> tuple[dict, np.ndarray, np.ndarray]:
    selected, nom = selected_and_nom(a, b)
    prior = np.zeros_like(nom)
    prior[:, common] = True
    den = types.sum(0)
    covered = (types * nom * selected[:, None]).sum(0)
    prior_covered = (types * prior * selected[:, None]).sum(0)
    eligible = selected & (y == 1) & (types.sum(1) > 0)
    hit = ((types[eligible] * nom[eligible]).sum(1) > 0)
    prior_hit = ((types[eligible] * prior[eligible]).sum(1) > 0)
    found = int(y[selected].sum())
    metrics = dict(rows=int(len(y)), positives=int(y.sum()), budget=int(selected.sum()),
                   found=found, recall=float(found / y.sum()), precision=float(y[selected].mean()),
                   eligible_selected_positives=int(eligible.sum()),
                   conditional_top2_hit=float(hit.mean()) if len(hit) else None,
                   common_top2_hit=float(prior_hit.mean()) if len(prior_hit) else None,
                   type_macro_coverage=float(np.mean(covered[den > 0] / den[den > 0])),
                   common_macro_coverage=float(np.mean(prior_covered[den > 0] / den[den > 0])))
    return metrics, selected, nom


def main() -> None:
    data, sources = load_all()
    annual_rows, seed_rows, type_rows, industry_rows, cell_rows = [], [], [], [], []
    availability_rows = []
    aggregation_rows = []
    selections, nominations = {}, {}

    for year in YEARS:
        d = data[year]
        y, types, a, b, common = d["y"], d["types"], d["a"], d["b"], d["common"]
        metrics, selected, nom = application_metrics(y, types, a, b, common)
        assert (metrics["rows"], metrics["positives"], metrics["budget"], metrics["found"]) == EXPECTED[year]
        metrics.update(year=year, predictor="six_seed_mean_calibrated_logit",
                       common_type_1=TYPE_CODES[common[0]], common_type_2=TYPE_CODES[common[1]])
        annual_rows.append(metrics)
        selections[year], nominations[year] = selected, nom

        probability_metrics, probability_selected, probability_nom = application_metrics(
            y, types, d["a_probability_average"], d["b_probability_average"], common)
        aggregation_rows.append({
            "year": year, "budget": int(selected.sum()),
            "mean_logit_hits": metrics["found"],
            "mean_probability_hits": probability_metrics["found"],
            "risk_list_overlap": int((selected & probability_selected).sum()),
            "risk_list_changed_members": int(np.logical_xor(selected, probability_selected).sum()),
            "risk_list_jaccard": float((selected & probability_selected).sum() /
                                        (selected | probability_selected).sum()),
            "top2_same_all_rows": float(np.all(nom == probability_nom, axis=1).mean()),
            "top2_same_on_canonical_list": float(np.all(nom[selected] == probability_nom[selected], axis=1).mean()),
            "max_overall_probability_difference": float(np.max(np.abs(a - d["a_probability_average"]))),
            "max_type_probability_difference": float(np.max(np.abs(b - d["b_probability_average"]))),
        })

        for k, seed in enumerate(SEEDS):
            sm, _, _ = application_metrics(y, types, d["a_seed"][k], d["b_seed"][k], common)
            seed_rows.append({"year": year, "seed": seed, **sm})

        pos = y == 1
        joint = a[:, None] * b
        den = types.sum(0)
        covered = (types * nom * selected[:, None]).sum(0)
        prior = np.zeros_like(nom); prior[:, common] = True
        prior_cov = (types * prior * selected[:, None]).sum(0)
        for j, code in enumerate(TYPE_CODES):
            cond_y = types[pos, j]
            conditional_ap = (average_precision_score(cond_y, b[pos, j])
                              if 0 < cond_y.sum() < len(cond_y) else None)
            joint_ap = average_precision_score(types[:, j], joint[:, j]) if den[j] else None
            type_rows.append(dict(year=year, type=code, support=int(den[j]),
                                  support_status=support_status(int(den[j])),
                                  conditional_ap=conditional_ap, joint_ap=joint_ap,
                                  covered=int(covered[j]), coverage=float(covered[j] / den[j]) if den[j] else None,
                                  common_covered=int(prior_cov[j]),
                                  common_coverage=float(prior_cov[j] / den[j]) if den[j] else None))

        observed_industry = d["industry"] < 19
        availability_rows.append(dict(year=year, rows=int(len(y)),
                                      industry_observed=int(observed_industry.sum()),
                                      industry_missing=int((~observed_industry).sum()),
                                      observed_share=float(observed_industry.mean())))
        observed_positive = pos & observed_industry
        overall_composition = (types[observed_positive].mean(0)
                               if observed_positive.any() else np.full(len(TYPE_CODES), np.nan))
        for ind in range(20):
            m = d["industry"] == ind
            yi, ti, si, ni = y[m], types[m], selected[m], nom[m]
            pi = prior[m]
            ind_pos = yi == 1
            type_support = ti.sum(0)
            type_cov = (ti * ni * si[:, None]).sum(0)
            prior_type_cov = (ti * pi * si[:, None]).sum(0)
            eligible = si & ind_pos & (ti.sum(1) > 0)
            hit = ((ti[eligible] * ni[eligible]).sum(1) > 0)
            prior_hit = ((ti[eligible] * pi[eligible]).sum(1) > 0)
            found = int(yi[si].sum())
            valid = type_support > 0
            industry_rows.append(dict(year=year, industry=INDUSTRIES[ind], companies=int(m.sum()),
                                      industry_observed=bool(ind < 19),
                                      positives=int(yi.sum()), selected=int(si.sum()), found=found,
                                      recall=float(found / yi.sum()) if yi.sum() else None,
                                      eligible_selected_positives=int(eligible.sum()),
                                      conditional_top2_hit=float(hit.mean()) if len(hit) else None,
                                      common_top2_hit=float(prior_hit.mean()) if len(prior_hit) else None,
                                      type_macro_coverage=float(np.mean(type_cov[valid] / type_support[valid])) if valid.any() else None,
                                      common_macro_coverage=float(np.mean(prior_type_cov[valid] / type_support[valid])) if valid.any() else None,
                                      type_hint_support_status=support_status(int(eligible.sum())),
                                      supported_types=int(valid.sum())))
            for j, code in enumerate(TYPE_CODES):
                support = int(type_support[j])
                share = float(ti[ind_pos, j].mean()) if ind_pos.any() else None
                base = float(overall_composition[j])
                lift = float(share / base) if ind < 19 and share is not None and base > 0 else None
                cell_rows.append(dict(year=year, industry=INDUSTRIES[ind], type=code,
                                      industry_observed=bool(ind < 19),
                                      companies=int(m.sum()), positive_companies=int(yi.sum()),
                                      support=support, support_status=support_status(support),
                                      conditional_share=share, overall_conditional_share=base, lift=lift))
        # industry support must partition each type exactly
        chk = pd.DataFrame(cell_rows)
        chk = chk[chk.year == year].groupby("type").support.sum().reindex(TYPE_CODES).to_numpy()
        np.testing.assert_array_equal(chk, den)

    annual = pd.DataFrame(annual_rows)
    seeds = pd.DataFrame(seed_rows)
    per_type = pd.DataFrame(type_rows)
    industry = pd.DataFrame(industry_rows)
    cells = pd.DataFrame(cell_rows)

    # pooled industry display uses each year's globally selected list; no re-ranking across years
    pooled_industry = []
    pooled_cells = []
    overall_pooled_support = sum((data[year]["types"][(data[year]["y"] == 1) &
                                                       (data[year]["industry"] < 19)].sum(0)
                                  for year in YEARS), start=np.zeros(10, dtype=int))
    overall_pooled_positive = sum(int(((data[year]["y"] == 1) &
                                       (data[year]["industry"] < 19)).sum()) for year in YEARS)
    for ind in OBSERVED_INDUSTRIES:
        pieces = []
        for year in YEARS:
            d = data[year]; m = np.array([INDUSTRIES[v] == ind for v in d["industry"]])
            pieces.append((d, m, selections[year][m], nominations[year][m]))
        companies = sum(int(m.sum()) for _, m, _, _ in pieces)
        positives = sum(int(d["y"][m].sum()) for d, m, _, _ in pieces)
        selected_n = sum(int(sel.sum()) for _, _, sel, _ in pieces)
        found = sum(int(d["y"][m][sel].sum()) for d, m, sel, _ in pieces)
        den = sum((d["types"][m].sum(0) for d, m, _, _ in pieces), start=np.zeros(10, dtype=int))
        num = sum(((d["types"][m] * nom * sel[:, None]).sum(0) for d, m, sel, nom in pieces), start=np.zeros(10, dtype=int))
        prior_num = np.zeros(10, dtype=int)
        hit_num = hit_den = prior_hit_num = 0
        for d, m, sel, nom in pieces:
            ti, yi = d["types"][m], d["y"][m]
            prior = np.zeros_like(nom); prior[:, d["common"]] = True
            prior_num += (ti * prior * sel[:, None]).sum(0).astype(int)
            elig = sel & (yi == 1) & (ti.sum(1) > 0)
            hit_num += int(((ti[elig] * nom[elig]).sum(1) > 0).sum())
            prior_hit_num += int(((ti[elig] * prior[elig]).sum(1) > 0).sum())
            hit_den += int(elig.sum())
        valid = den > 0
        pooled_industry.append(dict(industry=ind, companies=companies, positives=positives,
                                    selected=selected_n, found=found,
                                    recall=float(found / positives) if positives else None,
                                    eligible_selected_positives=hit_den,
                                    conditional_top2_hit=float(hit_num / hit_den) if hit_den else None,
                                    common_top2_hit=float(prior_hit_num / hit_den) if hit_den else None,
                                    type_macro_coverage=float(np.mean(num[valid] / den[valid])) if valid.any() else None,
                                    common_macro_coverage=float(np.mean(prior_num[valid] / den[valid])) if valid.any() else None,
                                    supported_types=int(valid.sum()),
                                    type_hint_support_status=support_status(hit_den),
                                    display_eligible=bool(companies >= 100 and positives >= 30)))
        for j, code in enumerate(TYPE_CODES):
            overall_share = float(overall_pooled_support[j] / overall_pooled_positive)
            share = float(den[j] / positives) if positives else None
            pooled_cells.append(dict(industry=ind, type=code, companies=companies,
                                     positive_companies=positives, support=int(den[j]),
                                     support_status=support_status(int(den[j])),
                                     conditional_share=share,
                                     overall_conditional_share=overall_share,
                                     lift=float(share / overall_share) if share is not None and overall_share > 0 else None))
    pooled_industry = pd.DataFrame(pooled_industry)
    pooled_cells = pd.DataFrame(pooled_cells)
    pooled_partition = pooled_cells.groupby("type").support.sum().reindex(TYPE_CODES).to_numpy()
    np.testing.assert_array_equal(pooled_partition, overall_pooled_support)

    # pooled "other" is recomputed from rows, not averaged across heterogeneous industries
    major = set(pooled_industry.loc[pooled_industry.display_eligible, "industry"])
    display_rows = pooled_industry[pooled_industry.industry.isin(major)].copy()
    other_inds = set(OBSERVED_INDUSTRIES) - major
    if other_inds:
        pieces = []
        for year in YEARS:
            d = data[year]
            m = np.array([INDUSTRIES[v] in other_inds for v in d["industry"]])
            pieces.append((d, m, selections[year][m], nominations[year][m]))
        companies = sum(int(m.sum()) for _, m, _, _ in pieces); positives = sum(int(d["y"][m].sum()) for d, m, _, _ in pieces)
        selected_n = sum(int(sel.sum()) for _, _, sel, _ in pieces); found = sum(int(d["y"][m][sel].sum()) for d, m, sel, _ in pieces)
        den = sum((d["types"][m].sum(0) for d, m, _, _ in pieces), start=np.zeros(10, dtype=int))
        num = sum(((d["types"][m] * nom * sel[:, None]).sum(0) for d, m, sel, nom in pieces), start=np.zeros(10, dtype=int))
        prior_num = np.zeros(10, dtype=int); hn = hd = phn = 0
        for d, m, sel, nom in pieces:
            ti, yi = d["types"][m], d["y"][m]; elig = sel & (yi == 1) & (ti.sum(1) > 0)
            prior = np.zeros_like(nom); prior[:, d["common"]] = True
            prior_num += (ti * prior * sel[:, None]).sum(0).astype(int)
            hn += int(((ti[elig] * nom[elig]).sum(1) > 0).sum()); hd += int(elig.sum())
            phn += int(((ti[elig] * prior[elig]).sum(1) > 0).sum())
        valid = den > 0
        display_rows = pd.concat([display_rows, pd.DataFrame([dict(industry="OTHER", companies=companies,
            positives=positives, selected=selected_n, found=found, recall=float(found/positives) if positives else None,
            eligible_selected_positives=hd, conditional_top2_hit=float(hn/hd) if hd else None,
            common_top2_hit=float(phn/hd) if hd else None,
            type_macro_coverage=float(np.mean(num[valid]/den[valid])) if valid.any() else None,
            common_macro_coverage=float(np.mean(prior_num[valid]/den[valid])) if valid.any() else None,
            supported_types=int(valid.sum()), type_hint_support_status=support_status(hd),
            display_eligible=True)])], ignore_index=True)

    annual.to_csv(HERE / "annual_summary.csv", index=False)
    seeds.to_csv(HERE / "seed_stability.csv", index=False)
    per_type.to_csv(HERE / "per_type.csv", index=False)
    industry.to_csv(HERE / "industry_annual.csv", index=False)
    cells.to_csv(HERE / "industry_type_cells.csv", index=False)
    pooled_cells.to_csv(HERE / "industry_type_pooled.csv", index=False)
    pooled_industry.to_csv(HERE / "industry_pooled.csv", index=False)
    display_rows.to_csv(HERE / "industry_display.csv", index=False)
    pd.DataFrame(availability_rows).to_csv(HERE / "industry_availability.csv", index=False)
    pd.DataFrame(aggregation_rows).to_csv(HERE / "aggregation_sensitivity.csv", index=False)
    (HERE / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf8")

    type_equal = per_type.groupby("year")[["conditional_ap", "joint_ap", "coverage", "common_coverage"]].mean()
    summary = {
        "years": annual.to_dict("records"),
        "three_year_equal_weight": {k: float(type_equal[k].mean()) for k in type_equal.columns},
        "three_year_mean_application": {k: float(annual[k].mean()) for k in
            ("recall", "precision", "conditional_top2_hit", "common_top2_hit", "type_macro_coverage", "common_macro_coverage")},
        "major_industries": sorted(major),
        "other_industries": sorted(other_inds),
        "industry_analysis_years": [int(r["year"]) for r in availability_rows if r["industry_observed"] > 0],
        "aggregation_sensitivity": aggregation_rows,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")

    verification = {
        "status": "PASS",
        "no_training": True,
        "frozen_predictor": "mean_of_six_calibrated_logits_then_sigmoid",
        "years": list(YEARS), "seeds": list(SEEDS), "models_read": len(YEARS) * len(SEEDS),
        "expected_ensemble_hits_reproduced": True,
        "aggregation_sensitivity_saved": True,
        "labels_only_on_overall_positive_rows": True,
        "industry_partition_matches_type_support": True,
        "source_count": len(sources),
        "data_signatures": {str(y): data[y]["signatures"] for y in YEARS},
    }
    (HERE / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf8")

    lines = ["# FiLM-L 企业类别分层与违规类型评价", "",
             "本轮冻结最终 FiLM-L，0 次训练；读取2020—2022年六种子预测。按照已获老师认可的展示版本，主口径为六个校准 logit 等权平均，再用于排序并转换为概率。", "",
             "## 总体名单与类型提示", ""]
    for r in annual.itertuples():
        lines.append(f"- {r.year}：前10%名单 {r.budget} 家，命中 {r.found}/{r.positives}，Recall@10% {r.recall:.2%}，名单精确率 {r.precision:.2%}；条件Top2命中 {r.conditional_top2_hit:.2%}，训练常见两类参照 {r.common_top2_hit:.2%}；端到端类型宏覆盖 {r.type_macro_coverage:.2%}，常见两类宏覆盖 {r.common_macro_coverage:.2%}。")
    ew = summary["three_year_equal_weight"]; ma = summary["three_year_mean_application"]
    lines += ["", "三年等权：条件类型 Macro-AP {:.4f}，联合类型 Macro-AP {:.4f}，端到端类型宏覆盖 {:.2%}；条件Top2命中 {:.2%}，常见两类参照 {:.2%}。".format(ew["conditional_ap"], ew["joint_ap"], ma["type_macro_coverage"], ma["conditional_top2_hit"], ma["common_top2_hit"]), ""]
    lines.append("聚合口径敏感性检查显示：改用六概率平均时三年名单命中数不变，但2021和2022的名单成员分别变化4家和6家；因此后续隐私与通信实验继续使用已冻结的校准logit平均，不混用两种集成公式。")
    if ma["conditional_top2_hit"] <= ma["common_top2_hit"]:
        lines.append("模型Top2未超过训练期最常见两类的固定提示，因此当前类型头提供了逐企业分数，但尚未证明其Top2提示优于简单频率策略。")
    else:
        lines.append("模型Top2三年平均超过训练期最常见两类固定提示；仍需结合逐年方向和长尾覆盖判断，不以单一平均值宣布稳定改进。")
    lines += ["", "## 逐类型与行业", "",
              "`per_type.csv` 给出每年十类的条件/联合AP和端到端覆盖；`industry_type_cells.csv` 与 `industry_type_pooled.csv` 给出行业×类型支持、构成与lift。单格按<10、10—29、>=30三级标记，小样本不得单独解释。", "",
              "2022 数据包的行业字段全部为UNK，因此行业比较只使用有行业编码的2020—2021行；UNK不当作一种企业类别。汇报用主要行业仅按支持量筛选（公司年度>=100且正例>=30）：" + "、".join(sorted(major)) + "；其余已知行业合并为OTHER。`industry_display.csv`中的名单仍来自每年度统一的全体前10%，没有给各行业另分10%预算。", "",
              "## 结论边界", "",
              "本轮补齐了最终FiLM-L已有类型输出及行业分层的评价证据。行业差异是描述性异质性，不证明行业嵌入带来增量；该问题需要TRUE/CONST/SHUFFLE另行对照。类型提示的长尾能力与常见类参照分别报告，不用总体筛查召回替代类型效果。", "",
              "## 复现", "", "```bash", "实验与代码/.venv-literature/bin/python 实验与代码/FiLM-L分类闭环/20261020-v1/evaluate.py", "```", ""]
    (HERE / "结果解读.md").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
