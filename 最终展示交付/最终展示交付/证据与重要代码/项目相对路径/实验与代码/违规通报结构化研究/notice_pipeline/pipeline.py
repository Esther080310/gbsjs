"""按记录调用、缓存和续跑；推断结果与原始标签分别保存。"""
from __future__ import annotations

import json
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .client import APIError, DeepSeekClient, parse_response, request_body
from .extraction import build_messages, validate_extraction
from .source import run_path, verify_run
from .storage import append_jsonl, digest, now, read_jsonl, repair_tail, write_json
from .recovery import partial_results, process_payload, recover


def body_for(cfg: dict, record: dict) -> dict:
    return request_body(cfg["api"], build_messages(record))


def valid_results(cfg: dict, records: list[dict]) -> dict[str, dict]:
    """只接受当前输入、当前请求和当前规则一致的已成功结果。"""
    known = {r["record_id"]: r for r in records}
    results = {}
    for row in read_jsonl(run_path(cfg) / "extractions.jsonl"):
        rid = row.get("record_id")
        if rid not in known:
            continue
        if row.get("request_hash") != digest(body_for(cfg, known[rid])):
            continue
        try:
            validate_extraction(row["extraction"], known[rid])
        except (ValueError, KeyError, TypeError):
            continue
        results[rid] = row
    return results


def inspect_run(cfg: dict) -> dict:
    manifest = verify_run(cfg)
    records = read_jsonl(run_path(cfg) / "prepared.jsonl")
    completed = valid_results(cfg, records)
    partial = {k: v for k, v in partial_results(cfg, records).items() if k not in completed}
    return {"run_dir": str(run_path(cfg)), "snapshot_cutoff": cfg["study"]["snapshot_cutoff"],
            "source_rows": manifest["source_rows"], "eligible_rows": manifest["eligible_rows"],
            "selected_rows": len(records), "completed": len(completed),
            "partial_records": len(partial),
            "pending": len(records) - len(completed) - len(partial),
            "not_fully_validated": len(records) - len(completed),
            "strata_counts": manifest["strata_counts"],
            "split_counts": manifest["split_counts"], "model": cfg["api"]["model"],
            "human_sampling": manifest["human_sampling"]}


def extract_one(cfg: dict, record: dict, client) -> dict:
    dest = run_path(cfg)
    body = body_for(cfg, record)
    base = {"record_id": record["record_id"], "request_hash": digest(body),
            "created_at": now(), "attempts": 0, "usage": {}}
    try:
        raw, attempts = client.complete(body)
        raw_file = Path("raw_responses") / (digest(record["record_id"])[:20] + "-" +
                                           uuid.uuid4().hex[:12] + ".json")
        write_json(dest / raw_file, raw)
        base.update(attempts=attempts, usage=raw.get("usage", {}),
                    response_model=raw.get("model"), response_id=raw.get("id"),
                    system_fingerprint=raw.get("system_fingerprint"),
                    raw_response_file=raw_file.as_posix())
        payload, status, recovery_meta = process_payload(parse_response(raw), record)
        count = sum(len(f["evidence"]) for f in payload["facts"])
        base.update(status=status, extraction=payload, recovery=recovery_meta,
                    validation={"passed": True, "evidence_count": count,
                                "complete_record": status == "success",
                                "validator_version": recovery_meta["validator_version"]})
    except APIError as exc:
        base.update(status="failure", error_kind="api", error=str(exc),
                    fatal=exc.fatal, attempts=exc.attempts, error_code=exc.error_code)
    except (ValueError, KeyError, TypeError) as exc:
        base.update(status="failure", error_kind="validation", error=str(exc), fatal=False)
    return base


def extract(cfg: dict, *, key: str = "", limit: int | None = None, dry_run: bool = False,
            retry_partial: bool = False, client=None, max_consecutive_api_failures: int = 5) -> dict:
    if max_consecutive_api_failures < 1:
        raise ValueError('连续接口失败停止阈值必须为正整数')
    verify_run(cfg)
    dest = run_path(cfg)
    records = read_jsonl(dest / "prepared.jsonl")
    for filename in ("extractions.jsonl", "failures.jsonl", "partials.jsonl"):
        repair_tail(dest / filename)
    if not dry_run:
        recover(cfg)  # 先复用已付费响应，再判断是否需要新请求。
    completed = valid_results(cfg, records)
    partial = {k: v for k, v in partial_results(cfg, records).items() if k not in completed}
    pending = [r for r in records if r["record_id"] not in completed
               and (retry_partial or r["record_id"] not in partial)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit必须为正整数")
        pending = pending[:limit]
    if dry_run:
        preview = {"network_requests": 0, "pending_this_call": len(pending),
                   "already_completed": len(completed),
                   "partial_records_cached": len(partial),
                   "input_characters": sum(len(json.dumps(body_for(cfg, r), ensure_ascii=False))
                                           for r in pending),
                   "note": "字符量含重复schema提示词；不是token或人民币报价。",
                   "first_request": body_for(cfg, pending[0]) if pending else None}
        write_json(dest / "request_preview.json", preview)
        return {k: v for k, v in preview.items() if k != "first_request"}
    if not pending:
        return {"new_success": 0, "new_failure": 0, "new_partial": 0,
                "completed_total": len(completed), "partial_records": len(partial), "pending": 0}
    if client is None:
        if not key.strip():
            raise ValueError("未设置DEEPSEEK_API_KEY。请使用extract --ask-key隐藏输入，或设置环境变量。")
        client = DeepSeekClient(cfg["api"], key.strip())
    counts = {"new_success": 0, "new_failure": 0, "new_partial": 0,
              "stopped_on_fatal_api_error": False,
              "stopped_on_repeated_api_errors": False}
    consecutive_api_failures = 0
    queue = iter(pending)
    workers = cfg["api"]["concurrency"]
    # 只维持并发数个在途请求，认证/余额等错误后停止提交后续记录。
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        for _ in range(min(workers, len(pending))):
            record = next(queue)
            active[executor.submit(extract_one, cfg, record, client)] = record["record_id"]
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                row = future.result()
                ok = row["status"] == "success"
                is_partial = row["status"] == "partial"
                filename = "extractions.jsonl" if ok else "partials.jsonl" if is_partial else "failures.jsonl"
                append_jsonl(dest / filename, row)
                counts["new_success" if ok else "new_partial" if is_partial else "new_failure"] += 1
                label = ("完成" if ok else f"部分完成（保留{len(row['extraction']['facts'])}项事实）"
                         if is_partial else "失败 " + row["error_kind"] + ": " + row["error"][:180])
                print(f"{row['record_id']}: {label}", flush=True)
                if row.get("fatal"):
                    counts["stopped_on_fatal_api_error"] = True
                if row.get('error_kind') == 'api':
                    consecutive_api_failures += 1
                else:
                    consecutive_api_failures = 0
                if consecutive_api_failures >= max_consecutive_api_failures:
                    counts['stopped_on_repeated_api_errors'] = True
            if not (counts["stopped_on_fatal_api_error"] or counts['stopped_on_repeated_api_errors']):
                while len(active) < workers:
                    record = next(queue, None)
                    if record is None:
                        break
                    active[executor.submit(extract_one, cfg, record, client)] = record["record_id"]
    counts["completed_total"] = len(valid_results(cfg, records))
    counts["partial_records"] = len(set(partial_results(cfg, records)) - set(valid_results(cfg, records)))
    counts["pending"] = len(records) - counts["completed_total"] - counts["partial_records"]
    if counts['stopped_on_repeated_api_errors']:
        print('连续接口失败，已停止提交新记录；已有响应保留，请先检查网络再续跑。', flush=True)
    write_json(dest / "last_execution.json", dict(counts, finished_at=now()))
    return counts
