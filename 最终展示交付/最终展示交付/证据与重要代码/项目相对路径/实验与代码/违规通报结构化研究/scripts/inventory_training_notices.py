"""只读盘点训练期通报调用候选；不调用API、不生成事实真值、不改标签。"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(P))

import numpy as np

from notice_pipeline.pipeline import body_for, valid_results
from notice_pipeline.recovery import partial_results
from notice_pipeline.source import (
    date_upper_bound, load_config, read_source, resolve, run_path, verify_run,
)
from notice_pipeline.storage import file_digest, read_jsonl, write_json

OUT = P / "artifacts/followup/training_notice_inventory"
PROTOCOLS = ("retrospective_snapshot2023", "asof_training_year_end")
VERSION = "training-notice-call-inventory-v1"


def symbol(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if re.fullmatch(r"\d{1,6}", text) else ""


def original_pairs(source: dict) -> tuple[list[dict], bool]:
    """保留原分号位置；不把整段案件的每个类型复制到所有年份。"""
    years = source["ViolationYear"].split(";")
    types = source["ViolationTypeID"].split(";")
    aligned = len(years) == len(types)
    pairs = []
    for i in range(max(len(years), len(types))):
        y = years[i] if i < len(years) else None
        t = types[i] if i < len(types) else None
        year = int(y.strip()) if y is not None and re.fullmatch(r"(?:19|20)\d{2}", y.strip()) else None
        pairs.append({"position": i, "year_literal": y, "year": year, "type_literal": t})
    return pairs, aligned


def text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def save_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checks() -> None:
    """直接检验会改变准入的边界，不要求网络或模型依赖。"""
    assert symbol("1") == "000001"
    assert symbol("000001") == "000001"
    assert symbol("1.0") == ""
    pairs, ok = original_pairs({"ViolationYear": "2015;N/A;2017", "ViolationTypeID": "P2501;P2599;P2505"})
    assert ok and [p["year"] for p in pairs] == [2015, None, 2017]
    assert [p["type_literal"] for p in pairs] == ["P2501", "P2599", "P2505"]
    assert not original_pairs({"ViolationYear": "2015;2016", "ViolationTypeID": "P2501"})[1]
    assert date_upper_bound("2017-00-00") == date(2017, 12, 31)
    assert date_upper_bound("2018-01-01") > date(2017, 12, 31)


def main() -> None:
    checks()
    cfg = load_config(P / "configs/pilot.toml")
    verify_run(cfg)
    run = run_path(cfg)
    y_path = resolve(cfg["source"]["path"])
    protected = [y_path, run / "prepared.jsonl", run / "extractions.jsonl", run / "partials.jsonl"]
    protected += [E / f"财务年度配对修正/cache/{ey}.npz" for ey in (2020, 2021)]
    before = {str(p): file_digest(p) for p in protected}
    records = read_source(y_path, cfg["source"].get("sheet", ""))
    prepared = read_jsonl(run / "prepared.jsonl")
    complete = valid_results(cfg, prepared)
    partial = {k: v for k, v in partial_results(cfg, prepared).items() if k not in complete}
    prepared_by_id = {r["record_id"]: r for r in prepared}
    holdout_groups = {r["document_group"] for r in prepared if r["split"] == "holdout"}
    known = {r["record_id"]: r for r in records}
    assert len(known) == len(records)
    for old in prepared:
        current = known[old["record_id"]]
        assert current["source"] == old["source"]
        assert current["document_group"] == old["document_group"]
        assert current["source_row"] == old["source_row"]

    indices, denominators, training_ends = {}, {}, {}
    for ey in (2020, 2021):
        index = {}
        with np.load(E / f"财务年度配对修正/cache/{ey}.npz", allow_pickle=False) as cache:
            for fold in range(4):
                symbols = cache[f"{fold}_symbol"]
                years = cache[f"{fold}_year"]
                labels = cache[f"{fold}_y"]
                assert len(symbols) == len(years) == len(labels)
                assert set(np.unique(labels)).issubset({0, 1})
                denominators[ey, fold] = {"samples": len(years), "positive_samples": int(labels.sum())}
                if fold == 0:
                    training_ends[ey] = date(int(years.max()), 12, 31)
                    assert training_ends[ey].year == ey - 3
                for s, y, label in zip(symbols, years, labels):
                    key = (symbol(s), int(y))
                    assert key[0] and key not in index
                    index[key] = (fold, int(label))
        indices[ey] = index

    cutoff = date.fromisoformat(cfg["study"]["snapshot_cutoff"])
    details = {}
    group_folds = {ey: defaultdict(set) for ey in (2020, 2021)}
    for record in records:
        rid, source = record["record_id"], record["source"]
        pairs, aligned = original_pairs(source)
        years = sorted({p["year"] for p in pairs if p["year"] is not None})
        sym = symbol(source["Symbol"])
        bound = date_upper_bound(source["DeclareDate"])
        reasons = []
        if source["IsViolated"].strip().upper() != "Y": reasons.append("listed_company_not_Y")
        if not source["Activity"].strip(): reasons.append("activity_empty")
        if bound is None: reasons.append("announcement_date_unknown_or_invalid")
        elif bound > cutoff: reasons.append("announcement_after_snapshot2023")
        if not sym: reasons.append("invalid_stock_symbol")
        if not aligned: reasons.append("original_year_type_positions_mismatch")
        if not years: reasons.append("no_explicit_original_year")
        details[rid] = dict(pairs=pairs, years=years, symbol=sym, bound=bound, reasons=reasons)
        # 同快照中同候选文书的任一公司年度位于非训练折，即标记跨折。
        # 这一隔离规则不限于IsViolated=Y的调用候选，避免漏掉同文书其他主体。
        if bound is not None and bound <= cutoff and sym and aligned:
            for ey in (2020, 2021):
                for year in years:
                    match = indices[ey].get((sym, year))
                    if match:
                        group_folds[ey][record["document_group"]].add(match[0])

    inventory, protocol_rows, coverage, exclusions = [], [], [], []
    strict_records = defaultdict(dict)
    request_chars = {}
    matched_rows_by_protocol = defaultdict(list)
    for record in records:
        rid, source, group = record["record_id"], record["source"], record["document_group"]
        info = details[rid]
        processed = "complete" if rid in complete else "partial" if rid in partial else "prepared_pending" if rid in prepared_by_id else "not_prepared"
        inventory.append(dict(
            record_id=rid, source_row=record["source_row"], document_group=group,
            symbol=info["symbol"], symbol_original=source["Symbol"], declare_date_original=source["DeclareDate"],
            declare_date_upper_bound=info["bound"].isoformat() if info["bound"] else "",
            is_violated_original=source["IsViolated"], original_violation_year=source["ViolationYear"],
            original_violation_type_id=source["ViolationTypeID"], original_year_type_pairs=text(info["pairs"]),
            activity_characters=len(source["Activity"]), punishment_characters=len(source.get("PunishmentMeasure", "")),
            source_text_characters=sum(len(str(v)) for v in source.values()),
            pilot_split=prepared_by_id.get(rid, {}).get("split", ""),
            pilot_holdout_document_group=group in holdout_groups, existing_processing=processed,
            base_candidate=not info["reasons"], base_exclusion_reasons=";".join(info["reasons"]),
            matched_fold0_positive_years_2020=text([year for year in info["years"] if indices[2020].get((info["symbol"], year)) == (0, 1)]),
            matched_fold0_positive_years_2021=text([year for year in info["years"] if indices[2021].get((info["symbol"], year)) == (0, 1)]),
            document_folds_2020=text(sorted(group_folds[2020][group])),
            document_folds_2021=text(sorted(group_folds[2021][group])),
        ))

    for ey in (2020, 2021):
        for protocol in PROTOCOLS:
            reason_counts = Counter()
            raw, strict, pending = [], [], []
            raw_keys, strict_keys, pending_keys = set(), set(), set()
            for record, inv in zip(records, inventory):
                rid, group = record["record_id"], record["document_group"]
                info = details[rid]
                matched = [year for year in info["years"] if indices[ey].get((info["symbol"], year)) == (0, 1)]
                reasons = list(info["reasons"])
                if not matched: reasons.append("no_matching_fold0_positive_company_year")
                if group in holdout_groups: reasons.append("pilot_holdout_document_group")
                folds = group_folds[ey][group]
                if 0 in folds and any(fold != 0 for fold in folds): reasons.append("candidate_document_spans_train_and_other_fold")
                if protocol == "asof_training_year_end":
                    if info["bound"] is None or info["bound"] > training_ends[ey]:
                        reasons.append("not_confirmed_public_by_training_year_end")
                for reason in reasons: reason_counts[reason] += 1
                if not reasons: reason_counts["eligible_strict_call_candidate"] += 1
                if info["reasons"] or not matched:
                    continue
                raw.append(record)
                raw_keys.update((info["symbol"], year) for year in matched)
                eligible = not reasons
                cached = rid in complete or rid in partial
                new_call = eligible and not cached
                row = dict(
                    evaluation_year=ey, protocol=protocol, training_publication_cutoff=training_ends[ey].isoformat() if protocol == "asof_training_year_end" else cutoff.isoformat(),
                    record_id=rid, source_row=record["source_row"], document_group=group,
                    symbol=info["symbol"], declare_date_original=record["source"]["DeclareDate"],
                    declare_date_upper_bound=info["bound"].isoformat(), original_violation_year=record["source"]["ViolationYear"],
                    original_violation_type_id=record["source"]["ViolationTypeID"],
                    original_year_type_pairs=text(info["pairs"]), matched_fold0_positive_years=text(matched),
                    document_matched_folds=text(sorted(folds)), pilot_split=inv["pilot_split"],
                    pilot_holdout_document_group=group in holdout_groups, existing_processing=inv["existing_processing"],
                    activity_characters=inv["activity_characters"], punishment_characters=inv["punishment_characters"],
                    source_text_characters=inv["source_text_characters"], strict_call_candidate=eligible,
                    requires_new_api_call=new_call, exclusion_reasons=";".join(reasons),
                    candidate_semantics="原年度定位的调用候选；不是已准入事实监督",
                )
                protocol_rows.append(row)
                matched_rows_by_protocol[ey, protocol].append(row)
                if eligible:
                    strict.append(record)
                    strict_keys.update((info["symbol"], year) for year in matched)
                    strict_records[rid][f"{ey}:{protocol}"] = matched
                    if new_call:
                        pending.append(record)
                        pending_keys.update((info["symbol"], year) for year in matched)
                        if rid not in request_chars:
                            request_chars[rid] = len(json.dumps(body_for(cfg, record), ensure_ascii=False))
            coverage.append(dict(
                evaluation_year=ey, protocol=protocol, publication_cutoff=training_ends[ey].isoformat() if protocol == "asof_training_year_end" else cutoff.isoformat(),
                fold0_samples=denominators[ey, 0]["samples"], fold0_positive_samples=denominators[ey, 0]["positive_samples"],
                base_matching_records=len(raw), base_matching_document_groups=len({r["document_group"] for r in raw}), base_matching_company_years=len(raw_keys),
                strict_candidate_records=len(strict), strict_candidate_document_groups=len({r["document_group"] for r in strict}), strict_candidate_company_years=len(strict_keys),
                cached_complete_records=sum(r["record_id"] in complete for r in strict), cached_partial_records=sum(r["record_id"] in partial for r in strict),
                new_call_candidate_records=len(pending), new_call_candidate_document_groups=len({r["document_group"] for r in pending}),
                new_call_candidate_company_years=len(pending_keys), new_call_activity_characters=sum(len(r["source"]["Activity"]) for r in pending),
                new_call_source_text_characters=sum(sum(len(str(v)) for v in r["source"].values()) for r in pending),
                new_call_serialized_request_characters=sum(request_chars[r["record_id"]] for r in pending),
                company_year_coverage_among_positive_samples=len(strict_keys) / denominators[ey, 0]["positive_samples"],
            ))
            for reason, count in sorted(reason_counts.items()):
                exclusions.append(dict(evaluation_year=ey, protocol=protocol, reason=reason, records=count,
                                       counting="排除原因可重叠；eligible_strict_call_candidate为互斥准入计数"))

    queue, cached_rows = [], []
    by_id = {r["record_id"]: r for r in inventory}
    for rid, protocols in sorted(strict_records.items()):
        inv = by_id[rid]
        row = {key: inv[key] for key in ("record_id", "source_row", "document_group", "symbol", "declare_date_original", "declare_date_upper_bound", "original_violation_year", "original_violation_type_id", "original_year_type_pairs", "pilot_split", "existing_processing", "activity_characters", "punishment_characters", "source_text_characters")}
        row.update(candidate_protocols=text(protocols), requires_new_api_call=rid not in complete and rid not in partial,
                   serialized_request_characters=request_chars.get(rid, 0),
                   candidate_semantics="调用前候选；抽取后仍须主体、行为、期间和原Y核验")
        (queue if row["requires_new_api_call"] else cached_rows).append(row)
    # 所有已完整/部分处理记录单独保存，含不进入训练扩批的pilot保留组。
    existing = [dict(r, successful_fact_count=len((complete.get(r["record_id"]) or partial.get(r["record_id"], {})).get("extraction", {}).get("facts", [])))
                for r in inventory if r["record_id"] in complete or r["record_id"] in partial]

    assert len(existing) == len(complete) + len(partial) == len(prepared)
    assert len({r["record_id"] for r in queue}) == len(queue)
    assert not set(r["record_id"] for r in queue) & (set(complete) | set(partial))
    assert not any(r["document_group"] in holdout_groups for r in queue)
    for row in protocol_rows:
        if row["strict_call_candidate"]:
            folds = json.loads(row["document_matched_folds"])
            assert folds == [0]
            assert json.loads(row["matched_fold0_positive_years"])
            if row["protocol"] == "asof_training_year_end":
                assert date.fromisoformat(row["declare_date_upper_bound"]) <= training_ends[row["evaluation_year"]]
    after = {str(p): file_digest(p) for p in protected}
    assert before == after, "原Y、原抽取日志或冻结缓存发生改变"

    OUT.mkdir(parents=True, exist_ok=True)
    save_csv("source_notice_inventory.csv", inventory, list(inventory[0]))
    save_csv("candidate_protocol_records.csv", protocol_rows, list(protocol_rows[0]))
    queue_fields = list(queue[0]) if queue else ["record_id", "source_row", "document_group", "requires_new_api_call"]
    save_csv("expansion_call_candidates.csv", queue, queue_fields)
    save_csv("eligible_cached_records.csv", cached_rows, queue_fields)
    save_csv("existing_processed_records.csv", existing, list(existing[0]))
    save_csv("protocol_coverage.csv", coverage, list(coverage[0]))
    save_csv("exclusion_counts.csv", exclusions, list(exclusions[0]))
    manifest = dict(
        version=VERSION, source_rows=len(records), source_sha256=before[str(y_path)],
        snapshot_cutoff=cutoff.isoformat(), pilot_records=len(prepared), complete_records=len(complete), partial_records=len(partial),
        pilot_holdout_document_groups=len(holdout_groups), new_call_candidates_unique=len(queue),
        new_call_candidate_document_groups=len({r["document_group"] for r in queue}),
        new_call_candidate_company_years=len({(r["symbol"], year) for r in queue for years in json.loads(r["candidate_protocols"]).values() for year in years}),
        new_call_activity_characters=sum(r["activity_characters"] for r in queue),
        new_call_source_text_characters=sum(r["source_text_characters"] for r in queue),
        new_call_serialized_request_characters=sum(r["serialized_request_characters"] for r in queue),
        coverage=coverage, original_year_type_alignment_counts=dict(Counter(str(original_pairs(r["source"])[1]) for r in records)),
        source_processing_counts=dict(Counter(r["existing_processing"] for r in inventory)),
        protected_inputs_sha256_before=before, protected_inputs_sha256_after=after,
        source_and_cache_unchanged=before == after, api_calls=0, training_runs=0,
        statement="只完成调用候选盘点；原ViolationYear匹配不代表事实已通过训练监督准入。",
        document_group_rule="沿用source.document_group候选键；非法律案件标识；检查同快照全部原始行在各年度缓存中的跨折映射。",
        char_budget_note="字符量不是token数或人民币费用；序列化请求含当前重复提示词/schema，未核对当前API价格。",
    )
    write_json(OUT / "manifest.json", manifest)
    lines = ["# 训练期通报扩批候选盘点", "", "本轮只读原Y、100条试点及冻结缓存，不调用API、不训练。", "",
             f"原Y共{len(records):,}条；现有{len(complete)}条完整处理、{len(partial)}条部分处理。候选清单已排除这些重复调用，并隔离pilot holdout候选文书组。", "",
             "## 原始年度匹配的用途", "",
             "ViolationYear与ViolationTypeID按原分号位置保存。只把明确四位年份与冻结缓存的证券代码—经济年度键匹配，要求fold0且原Y=1。这里定位的是值得调用的通报，不把通报中每项行为自动传播到所有原年份。抽取后仍须核验目标主体、独立违规行为、明确发生/报告期间、原Y及案件跨折。", "",
             "## 两套时间协议", "",
             "回顾性协议使用2023-12-31已公开通报；允许后续披露说明训练年度的历史行为。训练年度末前公开协议分别截至2017-12-31、2018-12-31。二者单独保存；后者也是保守时间约束，不代表已经重建了历史所有报表版本或历史标签。", "",
             "|评价年度|协议|训练正例|严格候选通报|候选企业年度|需新调用|新调用文书组|", "|---|---|---:|---:|---:|---:|---:|"]
    for row in coverage:
        lines.append(f"|{row['evaluation_year']}|{row['protocol']}|{row['fold0_positive_samples']}|{row['strict_candidate_records']}|{row['strict_candidate_company_years']}|{row['new_call_candidate_records']}|{row['new_call_candidate_document_groups']}|")
    lines += ["", "四行包含重复记录/企业年度，不能相加作为独立样本。去重合并后的新调用候选为"
              f"{manifest['new_call_candidates_unique']:,}条、{manifest['new_call_candidate_document_groups']:,}个候选文书组、{manifest['new_call_candidate_company_years']:,}个企业年度。正文Activity总字符量{manifest['new_call_activity_characters']:,}，全部原字段字符量{manifest['new_call_source_text_characters']:,}；按现提示词序列化请求总字符量{manifest['new_call_serialized_request_characters']:,}。这些量用于后续预算，未换算token或费用。", "",
              "## 文书隔离与复用", "",
              "沿用既有document_group候选键。在2023快照内所有原Y行中，任一同组公司年度同时进入训练折与其他折，即从该年度协议的严格候选中排除。该键可能合并或拆分真实案件；清单保留来源与匹配折，后续不能把候选组数量写成独立案件数。", "",
              "现有完整和部分处理记录均复用；部分处理记录的保留事实继续按原准入规则处理，本清单不触发重试。现有pilot holdout组保留用途，不能为了增加训练覆盖纳入本轮扩批。", "",
              "## 文件", "",
              "- source_notice_inventory.csv：原Y全部记录的原年类型串、分号位置、证券键、文本字符量、基础排除原因与缓存匹配。", "- candidate_protocol_records.csv：已通过基础条件且匹配训练正例的记录，逐年度/协议列出最终准入或排除原因。", "- expansion_call_candidates.csv：跨协议去重、去除已处理记录后的严格调用候选；不是API执行输入版本。", "- existing_processed_records.csv：已有100条处理状态、pilot split与保留事实数，供复用。", "- eligible_cached_records.csv：严格调用候选中已处理的记录，直接复用。", "- protocol_coverage.csv：各年度协议的候选覆盖；候选企业年度数不是实际可准入事实数。", "- exclusion_counts.csv：可重叠排除原因计数。", "- manifest.json：源文件哈希、统计和完整性检查。", "",
              "## 扩批前待固定", "",
              "先决定采用哪套时间协议及这次调用预算，再按候选文书组构造独立run_id和prepared版本；不改原100条split，也不自动运行全清单。抽取完成后重新执行事实—主体—期间准入，报告实际可用企业年度、行为区别和与原类型说明公平对照的共同掩码。", "",
              "原Y、两套NPZ缓存和原试点输入/日志的前后SHA256全部一致。未新增API请求或训练。"]
    (OUT / "盘点说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("source_rows", "complete_records", "partial_records", "new_call_candidates_unique", "new_call_candidate_document_groups", "new_call_candidate_company_years", "new_call_source_text_characters", "source_and_cache_unchanged", "api_calls")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
