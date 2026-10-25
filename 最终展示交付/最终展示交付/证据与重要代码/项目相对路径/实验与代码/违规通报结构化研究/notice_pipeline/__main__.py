"""python -m notice_pipeline prepare|inspect|extract|report|smoke。"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import unittest
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek违规事实与信息边界探索")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect", "extract", "recover", "report"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
        if name == "extract":
            command.add_argument("--limit", type=int)
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--ask-key", action="store_true", help="终端隐藏输入，不写入文件")
            command.add_argument("--retry-partial", action="store_true", help="明确重新调用已部分完成的记录")
    sub.add_parser("smoke", help="离线单元与集成检查；不调用API")
    args = parser.parse_args()
    if args.command == "smoke":
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent.parent / "tests"))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    from .source import load_config, prepare, run_path, verify_run
    from .pipeline import extract, inspect_run
    try:
        cfg = load_config(args.config)
        if args.command == "prepare":
            result = prepare(cfg)
            print(json.dumps({"run_dir": str(run_path(cfg)), "selected_rows": result["selected_rows"],
                              "eligible_rows": result["eligible_rows"],
                              "network_requests": 0}, ensure_ascii=False, indent=2))
        elif args.command == "inspect":
            print(json.dumps(inspect_run(cfg), ensure_ascii=False, indent=2))
        elif args.command == "recover":
            from .recovery import recover
            print(json.dumps(recover(cfg), ensure_ascii=False, indent=2))
        elif args.command == "extract":
            key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not args.dry_run:
                from .recovery import recover
                recover(cfg)
            if args.ask_key and not args.dry_run and (args.retry_partial or inspect_run(cfg)["pending"]):
                key = getpass.getpass("DeepSeek API Key（隐藏输入，不保存）: ")
            result = extract(cfg, key=key, limit=args.limit, dry_run=args.dry_run,
                             retry_partial=args.retry_partial)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result.get("new_failure", 0) else 0
        else:
            from .report import build_report
            verify_run(cfg)
            result = build_report(run_path(cfg))
            print(json.dumps({k: result[k] for k in (
                "prepared_records", "successful_records", "partial_records", "remaining_records", "facts",
                "partial_valid_facts", "isolated_facts", "unprocessed_records", "information_hypotheses",
                "source_label_pairs", "source_label_mismatch_records"
            )} | {"report": str(run_path(cfg) / "reports/report.md")}, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"操作未完成：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
