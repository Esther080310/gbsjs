"""Summarize bounded, evidence-checked notice extraction without changing Y."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .extraction import validate_extraction
from .storage import digest, read_jsonl


def _rows(path: Path, *, required: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    output = read_jsonl(path)
    if any(not isinstance(row, dict) for row in output):
        raise ValueError(f"{path}: JSONL 每行必须是对象")
    return output


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    # UTF-8 BOM lets macOS/Windows Excel recognize Chinese column values.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _counter(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"name": name, "occurrences": count} for name, count in counter.most_common()]


def _parts(value: Any) -> list[str]:
    # Keep N/A, zero month/day, ambiguous years and within-year type lists as-is.
    if value is None:
        return []
    return re.split(r"[;；]", str(value))


def _base(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source", {})
    return {
        "record_id": record["record_id"],
        "source_row": record.get("source_row", ""),
        "document_group": record.get("document_group", ""),
        "split": record.get("split", ""),
        "strata_json": _json(record.get("strata", [])),
        "ViolationID": source.get("ViolationID", ""),
        "Symbol": source.get("Symbol", ""),
        "CoFullName": source.get("CoFullName", ""),
        "DeclareDate": source.get("DeclareDate", ""),
        "DisposalDate": source.get("DisposalDate", ""),
    }


def build_report(run_dir: Path) -> dict[str, Any]:
    """Write reports for the prepared set, retaining the last successful result.

    Evidence pass rates are literal-reference engineering checks. They are not
    semantic extraction accuracy, label accuracy, or future prediction accuracy.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = _rows(run_dir / "prepared.jsonl", required=True)
    prepared: dict[str, dict[str, Any]] = {}
    for record in records:
        rid = record.get("record_id")
        if not isinstance(rid, str) or not rid:
            raise ValueError("prepared.jsonl 存在空或非字符串 record_id")
        if rid in prepared:
            raise ValueError(f"prepared.jsonl record_id 重复: {rid}")
        if not isinstance(record.get("source"), dict):
            raise ValueError(f"prepared.jsonl source 必须是对象: {rid}")
        prepared[rid] = record

    success_rows = _rows(run_dir / "extractions.jsonl")
    successes: dict[str, dict[str, Any]] = {}
    excluded_successes = 0
    excluded_request_mismatches = 0
    duplicates = 0
    for entry in success_rows:
        rid = entry.get("record_id")
        if rid not in prepared:
            excluded_successes += 1
            continue
        if manifest.get("config"):
            from .pipeline import body_for
            if entry.get("request_hash") != digest(body_for(manifest["config"], prepared[rid])):
                excluded_request_mismatches += 1
                continue
        extraction = entry.get("extraction")
        if not isinstance(extraction, dict):
            raise ValueError(f"成功记录缺少 extraction 对象: {rid}")
        if extraction.get("record_id") != rid:
            raise ValueError(f"外层和 extraction 内层 record_id 不一致: {rid}")
        validation = entry.get("validation", {})
        if isinstance(validation, dict) and validation.get("passed") is False:
            raise ValueError(f"extractions.jsonl 包含校验失败结果: {rid}")
        entry = dict(entry, extraction=validate_extraction(extraction, prepared[rid]))
        duplicates += int(rid in successes)
        successes[rid] = entry

    failure_rows = _rows(run_dir / "failures.jsonl")
    failures = [item for item in failure_rows if item.get("record_id") in prepared]
    unresolved_failed = set(item["record_id"] for item in failures) - successes.keys()
    from .recovery import partial_results
    partials = ({k: v for k, v in partial_results(manifest["config"], records).items()
                 if k not in successes} if manifest.get("config") else {})
    base_fields = list(_base(next(iter(prepared.values())))) if prepared else [
        "record_id", "source_row", "document_group", "split", "strata_json",
        "ViolationID", "Symbol", "CoFullName", "DeclareDate", "DisposalDate",
    ]
    fact_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    mechanisms: Counter[str] = Counter()
    holders: Counter[str] = Counter()
    split_prepared: Counter[str] = Counter()
    split_success: Counter[str] = Counter()
    strata_prepared: Counter[str] = Counter()
    strata_success: Counter[str] = Counter()
    evidence_count = quote_pass = offset_pass = evidence_pass = 0
    fact_reference_pass = 0
    factless_records = 0
    total_usage: Counter[str] = Counter()
    evidence_issues: list[dict[str, Any]] = []

    for rid, record in prepared.items():
        base = _base(record)
        source = record["source"]
        split = str(record.get("split", ""))
        split_prepared[split] += 1
        strata = record.get("strata", [])
        for name in strata:
            strata_prepared[str(name)] += 1

        years = _parts(source.get("ViolationYear"))
        type_names = _parts(source.get("ViolationType"))
        type_ids = _parts(source.get("ViolationTypeID"))
        lengths = {"years": len(years), "types": len(type_names), "type_ids": len(type_ids)}
        aligned = len(set(lengths.values())) == 1 and lengths["years"] > 0
        for index, (year, names, ids) in enumerate(
            zip_longest(years, type_names, type_ids, fillvalue=None), 1
        ):
            label_rows.append({
                **base, "pair_index": index,
                "year_original": year if year is not None else "",
                "violation_type_original": names if names is not None else "",
                "violation_type_id_original": ids if ids is not None else "",
                "pairing_status": "source_positions_aligned" if aligned else "source_length_mismatch",
                "source_segment_lengths_json": _json(lengths),
                "IsViolated": source.get("IsViolated", ""),
                "ViolationYear_full_original": source.get("ViolationYear", ""),
                "ViolationType_full_original": source.get("ViolationType", ""),
                "ViolationTypeID_full_original": source.get("ViolationTypeID", ""),
            })
        if rid not in successes:
            continue
        entry = successes[rid]
        extraction = entry["extraction"]
        split_success[split] += 1
        for name in strata:
            strata_success[str(name)] += 1
        facts = extraction.get("facts", [])
        hypotheses = extraction.get("information_hypotheses", [])
        if not isinstance(facts, list) or not isinstance(hypotheses, list):
            raise ValueError(f"facts/information_hypotheses 必须是数组: {rid}")
        factless_records += int(not facts)
        usage = entry.get("usage", {})
        if isinstance(usage, dict):
            for name, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    total_usage[name] += value
        for index, fact in enumerate(facts, 1):
            if not isinstance(fact, dict):
                raise ValueError(f"fact 必须是对象: {rid}/{index}")
            mechanism = fact.get("mechanism", "")
            if isinstance(mechanism, str) and mechanism:
                mechanisms[mechanism] += 1
            evidence = fact.get("evidence", [])
            if not isinstance(evidence, list):
                raise ValueError(f"evidence 必须是数组: {rid}/{index}")
            all_references_passed = bool(evidence)
            for citation in evidence:
                evidence_count += 1
                field = citation.get("field") if isinstance(citation, dict) else None
                quote = citation.get("quote") if isinstance(citation, dict) else None
                text = source.get(field) if isinstance(field, str) else None
                literal_ok = isinstance(text, str) and isinstance(quote, str) and bool(quote) and quote in text
                start = citation.get("start") if isinstance(citation, dict) else None
                end = citation.get("end") if isinstance(citation, dict) else None
                offsets_ok = bool(
                    literal_ok and isinstance(start, int) and not isinstance(start, bool)
                    and isinstance(end, int) and not isinstance(end, bool)
                    and 0 <= start < end <= len(text) and text[start:end] == quote
                )
                quote_pass += int(literal_ok)
                offset_pass += int(offsets_ok)
                passed = literal_ok and offsets_ok
                evidence_pass += int(passed)
                all_references_passed = all_references_passed and passed
                if not passed:
                    evidence_issues.append({"record_id": rid, "fact_id": fact.get("fact_id"), "citation": citation})
            fact_reference_pass += int(all_references_passed)
            fact_rows.append({
                **base, "fact_index": index, "fact_id": fact.get("fact_id", ""),
                "actor_json": _json(fact.get("actor", {})),
                "behavior": fact.get("behavior", ""), "mechanism": mechanism,
                "time_json": _json(fact.get("time", {})),
                "amounts_json": _json(fact.get("amounts", [])),
                "accounts_json": _json(fact.get("accounts", [])),
                "evidence_json": _json(evidence), "evidence_count": len(evidence),
                "all_evidence_literal_offsets_passed": all_references_passed,
                "fact_json": _json(fact), "request_hash": entry.get("request_hash", ""),
            })
        for index, hypothesis in enumerate(hypotheses, 1):
            if not isinstance(hypothesis, dict):
                raise ValueError(f"information_hypothesis 必须是对象: {rid}/{index}")
            candidate_holders = hypothesis.get("candidate_holders", [])
            if isinstance(candidate_holders, list):
                for holder in candidate_holders:
                    holders[holder if isinstance(holder, str) else _json(holder)] += 1
            hypothesis_rows.append({
                **base, "hypothesis_index": index,
                "fact_ids_json": _json(hypothesis.get("fact_ids", [])),
                "candidate_holders_json": _json(candidate_holders),
                "needed_observations_json": _json(hypothesis.get("needed_observations", [])),
                "cross_check": hypothesis.get("cross_check", ""),
                "basis": hypothesis.get("basis", ""),
                "hypothesis_json": _json(hypothesis), "request_hash": entry.get("request_hash", ""),
            })

    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    partial_fact_rows, partial_hypothesis_rows, issue_rows = [], [], []
    for rid, entry in partials.items():
        base = _base(prepared[rid])
        meta = entry.get("recovery", {})
        for fact in entry["extraction"]["facts"]:
            partial_fact_rows.append({**base, "fact_id": fact["fact_id"], "fact_json": _json(fact),
                                      "raw_response_file": entry.get("raw_response_file", "")})
        for hypothesis in entry["extraction"]["information_hypotheses"]:
            partial_hypothesis_rows.append({**base, "hypothesis_json": _json(hypothesis)})
        for issue in meta.get("rejected_facts", []):
            issue_rows.append({**base, "fact_id": issue["fact_id"], "error": issue["error"],
                               "fact_json": _json(issue["fact"])})
    _csv(reports / "partial_facts.csv", partial_fact_rows,
         base_fields + ["fact_id", "fact_json", "raw_response_file"])
    _csv(reports / "partial_hypotheses.csv", partial_hypothesis_rows, base_fields + ["hypothesis_json"])
    _csv(reports / "partial_issues.csv", issue_rows, base_fields + ["fact_id", "error", "fact_json"])
    _csv(reports / "facts.csv", fact_rows, base_fields + [
        "fact_index", "fact_id", "actor_json", "behavior", "mechanism", "time_json", "amounts_json",
        "accounts_json", "evidence_json", "evidence_count", "all_evidence_literal_offsets_passed",
        "fact_json", "request_hash",
    ])
    _csv(reports / "hypotheses.csv", hypothesis_rows, base_fields + [
        "hypothesis_index", "fact_ids_json", "candidate_holders_json", "needed_observations_json",
        "cross_check", "basis", "hypothesis_json", "request_hash",
    ])
    _csv(reports / "source_label_pairs.csv", label_rows, base_fields + [
        "pair_index", "year_original", "violation_type_original", "violation_type_id_original",
        "pairing_status", "source_segment_lengths_json", "IsViolated", "ViolationYear_full_original",
        "ViolationType_full_original", "ViolationTypeID_full_original",
    ])
    count = len(prepared)
    summary = {
        "run_id": manifest.get("run_id", run_dir.name),
        "manifest": manifest,
        "prepared_records": count,
        "prepared_document_groups": len({record.get("document_group") for record in records}),
        "successful_records": len(successes),
        "partial_records": len(partials),
        "partial_valid_facts": len(partial_fact_rows),
        "partial_valid_hypotheses": len(partial_hypothesis_rows),
        "isolated_facts": len(issue_rows),
        "unprocessed_records": count - len(successes) - len(partials),
        "offline_recovered_complete_records": sum("recovered_at" in r for r in successes.values()),
        "extraction_coverage": len(successes) / count if count else None,
        "remaining_records": count - len(successes),
        "failure_attempts_in_prepared_set": len(failures),
        "unresolved_failed_records": len(unresolved_failed),
        "duplicate_success_rows_discarded": duplicates,
        "out_of_prepared_success_rows_excluded": excluded_successes,
        "mismatched_request_success_rows_excluded": excluded_request_mismatches,
        "factless_successful_records": factless_records,
        "facts": len(fact_rows), "information_hypotheses": len(hypothesis_rows),
        "source_label_pairs": len(label_rows),
        "source_label_mismatch_records": len({row["record_id"] for row in label_rows if row["pairing_status"] != "source_positions_aligned"}),
        "split_coverage": [{"split": name, "prepared": n, "successful": split_success[name]} for name, n in sorted(split_prepared.items())],
        "strata_coverage": [{"stratum": name, "prepared": n, "successful": strata_success[name]} for name, n in sorted(strata_prepared.items())],
        "candidate_mechanism_occurrences": _counter(mechanisms),
        "candidate_holder_occurrences": _counter(holders),
        "evidence_checks": {
            "citations": evidence_count, "literal_quote_passed": quote_pass,
            "literal_offset_passed": offset_pass, "passed": evidence_pass,
            "pass_rate": evidence_pass / evidence_count if evidence_count else None,
            "facts_with_all_references_passed": fact_reference_pass,
            "issues": evidence_issues,
            "meaning": "引用属于源字段且字符偏移对应的工程校验；不评价语义正确性、标签准确性或预测准确率。",
        },
        "selected_success_usage": dict(total_usage),
        "usage_scope": "仅按每条记录最后成功结果汇总；重试、失败及被替换成功调用的用量不包含在内，不等于总账单。",
        "source_label_policy": "源年份/类型/编码按分号位置保留，不由LLM改写；长度不一致显式标记。",
        "semantic_validation": "prior_user_validation_accepted_no_new_manual_sampling",
    }
    (reports / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def percent(value: float | None) -> str:
        return "尚无可计算分母" if value is None else f"{value:.2%}"

    lines = [
        "# 违规通报结构化小试验报告", "",
        f"运行版本：`{summary['run_id']}`。以下数字来自本运行目录，不代替下游违规识别实验。", "",
        "## 1. 实际处理进度", "",
        f"准备 {count} 条记录、{summary['prepared_document_groups']} 个文书组；完整通过 {len(successes)} 条，占 {percent(summary['extraction_coverage'])}；未完整通过 {summary['remaining_records']} 条。",
        f"另有 {len(partials)} 条为部分完成，保留 {len(partial_fact_rows)} 项通过校验的事实，{len(issue_rows)} 项问题事实单独隔离；尚未处理 {summary['unprocessed_records']} 条。部分完成不计入完整通过率。",
        f"保留 {len(failures)} 条历史失败日志；其中 {summary['offline_recovered_complete_records']} 条记录已离线恢复为完整结果。历史日志保持原样，当前进度以上述完整、部分、未处理三类为准。",
        f"成功结果中 {factless_records} 条没有可抽取事实；共形成 {len(fact_rows)} 项事实、{len(hypothesis_rows)} 项信息需求假设。", "",
        "## 2. 引用工程校验", "",
        f"校验 {evidence_count} 段引用，字面引用与偏移同时通过 {evidence_pass} 段，通过率 {percent(summary['evidence_checks']['pass_rate'])}。",
        "本轮沿用用户确认的既有LLM人工验证，跳过新的人工抽检；此处比例只表示原文引用与字符位置通过程序检查。", "",
        "## 3. 候选机制", "",
    ]
    lines.extend(f"- {row['name']}：{row['occurrences']} 项事实。" for row in summary["candidate_mechanism_occurrences"])
    if not mechanisms:
        lines.append("本运行尚无候选机制计数。")
    lines.extend(["", "## 4. 候选信息持有方", ""])
    lines.extend(f"- {row['name']}：{row['occurrences']} 次候选提及。" for row in summary["candidate_holder_occurrences"])
    if not holders:
        lines.append("本运行尚无候选持有方计数。")
    lines.extend([
        "", "上述计数按模型原文统计，未合并同义词；它们是研究候选项，不是机构已持有、允许共享或同意合作的证明。分层小样本中的频次也不代表全部违规的发生比例。", "",
        "离线恢复补充的引用来自同一记录的原始文本，用于定位上下文。字段能够定位与其语义归属是不同的检查层次；部分完成表按事实隔离问题项，保留原输出与原因，不混入完整结果统计。", "",
        "## 5. 标签与时间边界", "",
        f"原始年度—类型位置配对共 {len(label_rows)} 项；{summary['source_label_mismatch_records']} 条记录的源段数不一致，已在 CSV 标记。",
        "模糊年份、N/A 与含 00 的日期保持原值；通报事实用于形成监督或研究假设，评分时只能使用评分日期之前可获得的特征。",
        "development/holdout 是按文书组预先划分的保留范围。自动解析全部记录不形成独立人工语义测试，也不产生泛化准确率。", "",
        "## 6. 结果文件", "",
        "- `summary.json`：覆盖、分层、机制、持有方、工程校验和成功调用用量。",
        "- `facts.csv`：逐事实内容、时间、金额和可定位引用；完整嵌套项保存在 JSON 列。",
        "- `hypotheses.csv`：逐项候选信息需求，保持与事实 ID 的关联。",
        "- `source_label_pairs.csv`：全部准备记录的源标签位置配对，不受模型抽取成败影响。", "",
        "- `partial_facts.csv`、`partial_hypotheses.csv`：部分完成记录中通过同一校验的事实与候选；不混入完整通过表。",
        "- `partial_issues.csv`：未通过事实及原输出，保留问题原因。原始API响应与既有失败日志保持不变。", "",
        "## 7. 下一步判断", "",
        "优先查看反复出现且确需独立机构交叉印证的候选事实，逐项核对可获得字段、主体范围、时间和授权用途。完成制度与数据核对后，才进入机制模拟和预测实验。",
        "本轮按用户要求跳过人工抽检；工程校验已单列，当前输出可供研究候选筛选，不自动覆盖正式 Y 或最终模型。", "",
    ])
    (reports / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
