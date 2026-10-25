"""小批量JSON文件工具；日志只保存研究数据，不保存认证信息。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value) -> None:
    # 由主线程写；断点期间可能留下半行，下次读取时仅忽略最后一个不完整行。
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.seek(0, 2)
                stream.write(b"\n")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines()
    result = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1 and not data.endswith(b"\n"):
                # 保留原文件，以便查验；续跑前由repair_tail修复这个末尾。
                break
            raise ValueError(f"JSONL损坏：{path.name} 第{i + 1}行") from None
    return result


def repair_tail(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    last = data.rsplit(b"\n", 1)[-1]
    try:
        json.loads(last)
    except (json.JSONDecodeError, UnicodeDecodeError):
        path.with_suffix(path.suffix + ".interrupted").write_bytes(last)
        prefix = data[:len(data) - len(last)]
        path.write_bytes(prefix)
