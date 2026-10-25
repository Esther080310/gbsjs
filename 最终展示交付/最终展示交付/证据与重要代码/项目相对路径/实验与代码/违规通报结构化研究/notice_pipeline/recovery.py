"""复用已付费响应；仅补充可定位引用及无歧义格式，隔离仍不合约的事实。"""
from __future__ import annotations

import copy
import json
from collections import Counter

from .client import parse_response
from .extraction import EVIDENCE_FIELDS, VALIDATOR_VERSION, validate_extraction
from .storage import append_jsonl, digest, now, read_jsonl, repair_tail, write_json

RECOVERY_VERSION = "evidence-closure-v1"


def _canonical(text: str) -> tuple[str, list[int]]:
    # 只统一全半角、弯引号和空白；不删数字、正负号、小数点或百分号。
    chars, positions = [], []
    mapping = {"“": '"', "”": '"', "‘": "'", "’": "'"}
    for i, char in enumerate(text):
        normalized_char = mapping.get(char, char)
        if 0xFF01 <= ord(char) <= 0xFF5E:
            normalized_char = chr(ord(char) - 0xFEE0)
        for normalized in normalized_char:
            if not normalized.isspace():
                chars.append(normalized)
                positions.append(i)
    return "".join(chars), positions


def _locate(texts: dict[str, str], literal: str, preferred: str | None = None):
    fields = ([preferred] if preferred in texts else []) + [f for f in texts if f != preferred]
    for field in fields:
        start = texts[field].find(literal)
        if start >= 0:
            return field, start, start + len(literal)
    return None


def normalize_payload(payload: dict, record: dict) -> tuple[dict, list[dict]]:
    """不改变事实内容/年份/数值；把已有来源信息补为引用，保留每次变更。"""
    output = copy.deepcopy(payload)
    changes = []
    texts = {f: str(record["source"].get(f) or "") for f in EVIDENCE_FIELDS}
    for fact in output.get("facts", []):
        fid = fact.get("fact_id")
        for evidence in fact.get("evidence", []):
            quote, original_field = evidence.get("quote"), evidence.get("field")
            if not isinstance(quote, str) or not quote:
                continue
            located = _locate(texts, quote, original_field)
            if located is None:
                target, _ = _canonical(quote)
                if target:
                    for field, text in texts.items():
                        normalized, positions = _canonical(text)
                        start = normalized.find(target)
                        if start >= 0:
                            located = field, positions[start], positions[start + len(target) - 1] + 1
                            break
            if located is not None:
                field, start, end = located
                corrected = texts[field][start:end]
                if field != original_field or corrected != quote:
                    changes.append({"fact_id": fid, "action": "literal_quote_normalization",
                                    "before": dict(evidence), "after": {"field": field, "quote": corrected}})
                    evidence.clear()
                    evidence.update(field=field, quote=corrected)
        values = [("actor.name", fact.get("actor", {}).get("name")),
                  ("time.literal", fact.get("time", {}).get("literal"))]
        values += [(f"amounts[{i}].literal", a.get("literal"))
                   for i, a in enumerate(fact.get("amounts", []))]
        values += [(f"accounts[{i}]", a) for i, a in enumerate(fact.get("accounts", []))]
        for path, literal in values:
            if not isinstance(literal, str) or not literal:
                continue
            if any(literal in str(e.get("quote", "")) for e in fact.get("evidence", [])):
                continue
            located = _locate(texts, literal)
            if located is None:
                continue  # 不猜姓名、不补年、不伪造原文；交给严格校验/隔离。
            field, start, end = located
            # 保存限定上下文，方便理解名称/期间所在位置；原值不改。
            left, right = max(0, start - 40), min(len(texts[field]), end + 60)
            evidence = {"field": field, "quote": texts[field][left:right]}
            fact.setdefault("evidence", []).append(evidence)
            changes.append({"fact_id": fid, "action": "add_context_reference",
                            "path": path, "value": literal, "source_field": field,
                            "start": start, "end": end})
        time = fact.get("time", {})
        if (time.get("status") == "not_mentioned" and time.get("literal") is None
                and time.get("years") is None and time.get("role") != "unknown"):
            changes.append({"fact_id": fid, "action": "normalize_empty_time_role",
                            "before": time.get("role"), "after": "unknown"})
            time["role"] = "unknown"
    return output, changes


def process_payload(payload: dict, record: dict) -> tuple[dict, str, dict]:
    """完整通过或明确标记部分通过；不删除问题原值，不把部分结果叫完整成功。"""
    normalized, changes = normalize_payload(payload, record)
    meta = {"version": RECOVERY_VERSION, "validator_version": VALIDATOR_VERSION,
            "changes": changes, "rejected_facts": [], "rejected_hypotheses": []}
    try:
        return validate_extraction(normalized, record), "success", meta
    except ValueError as exc:
        meta["whole_record_error"] = str(exc)
    accepted = []
    for fact in normalized.get("facts", []):
        candidate = {"record_id": normalized.get("record_id"), "facts": [fact],
                     "information_hypotheses": [], "extraction_note": normalized.get("extraction_note")}
        try:
            checked = validate_extraction(candidate, record)
            accepted.extend(checked["facts"])
        except ValueError as exc:
            meta["rejected_facts"].append({"fact_id": fact.get("fact_id"), "error": str(exc),
                                            "fact": fact})
    ids = {f["fact_id"] for f in accepted}
    hypotheses = []
    for hypothesis in normalized.get("information_hypotheses", []):
        if not set(hypothesis.get("fact_ids", [])).issubset(ids):
            meta["rejected_hypotheses"].append(hypothesis)
            continue
        candidate = {"record_id": normalized.get("record_id"), "facts": accepted,
                     "information_hypotheses": [hypothesis],
                     "extraction_note": normalized.get("extraction_note")}
        try:
            validate_extraction(candidate, record)
            hypotheses.append(hypothesis)
        except ValueError:
            meta["rejected_hypotheses"].append(hypothesis)
    # 保留顶层契约，不能用隔离流程掩盖record_id或额外字段问题。
    partial = dict(normalized, facts=accepted, information_hypotheses=hypotheses)
    if not accepted:
        partial["extraction_note"] = "本地校验未得到可完整定位的事实；原始输出及问题项保存在recovery记录。"
    return validate_extraction(partial, record), "partial", meta


def partial_results(cfg: dict, records: list[dict]) -> dict[str, dict]:
    from .pipeline import body_for
    from .source import run_path
    known = {r["record_id"]: r for r in records}
    result = {}
    for row in read_jsonl(run_path(cfg) / "partials.jsonl"):
        rid = row.get("record_id")
        if rid not in known or row.get("request_hash") != digest(body_for(cfg, known[rid])):
            continue
        try:
            validate_extraction(row["extraction"], known[rid])
        except (ValueError, KeyError, TypeError):
            continue
        result[rid] = row
    return result


def recover(cfg: dict) -> dict:
    from .pipeline import body_for, valid_results
    from .source import run_path, verify_run
    verify_run(cfg)
    dest = run_path(cfg)
    records = read_jsonl(dest / "prepared.jsonl")
    known = {r["record_id"]: r for r in records}
    for filename in ("extractions.jsonl", "partials.jsonl"):
        repair_tail(dest / filename)
    completed = valid_results(cfg, records)
    partials = partial_results(cfg, records)
    latest = {}
    for row in read_jsonl(dest / "failures.jsonl"):
        rid = row.get("record_id")
        if rid not in known or rid in completed or rid in partials:
            continue
        if row.get("request_hash") != digest(body_for(cfg, known[rid])):
            continue
        if row.get("raw_response_file"):
            latest[rid] = row
    counts = Counter()
    for rid, row in latest.items():
        try:
            raw = json.loads((dest / row["raw_response_file"]).read_text(encoding="utf-8"))
            payload, status, meta = process_payload(parse_response(raw), known[rid])
        except (ValueError, KeyError, TypeError) as exc:
            counts["still_failed"] += 1
            continue
        recovered = {k: v for k, v in row.items() if k not in {"error", "error_kind", "fatal", "status"}}
        recovered.update(status=status, extraction=payload, recovery=meta,
                         recovered_at=now(), validation={"passed": True,
                         "validator_version": VALIDATOR_VERSION,
                         "complete_record": status == "success",
                         "evidence_count": sum(len(f["evidence"]) for f in payload["facts"])})
        append_jsonl(dest / ("extractions.jsonl" if status == "success" else "partials.jsonl"), recovered)
        counts["recovered_complete" if status == "success" else "recovered_partial"] += 1
    complete = valid_results(cfg, records)
    partial = {k: v for k, v in partial_results(cfg, records).items() if k not in complete}
    result = dict(counts, network_requests=0, completed_total=len(complete), partial_records=len(partial),
                  pending=len(records) - len(complete) - len(partial))
    write_json(dest / "recovery_summary.json", dict(result, finished_at=now()))
    return result
