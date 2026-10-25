"""Freeze and encode B/C auxiliary targets locally; never train or call an LLM API.

The rows of B_embeddings/C_embeddings are aligned with targets.jsonl's
target_index. A single exact-text pool is encoded once for both conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
MAX_TOKENS = 512
DIMENSION = 512
P = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = P / "artifacts/followup/supervision_targets/targets.jsonl"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def library_versions() -> dict:
    result = {"python": platform.python_version()}
    for name in (
        "torch", "numpy", "transformers", "tokenizers", "huggingface-hub",
        "safetensors", "requests", "packaging", "regex", "filelock", "PyYAML",
        "tqdm", "fsspec", "hf-xet",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def read_targets(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("targets.jsonl has no targets")
    if [r["target_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("target_index must be unique, contiguous and in file order")
    if len({r["target_id"] for r in rows}) != len(rows):
        raise ValueError("target_id must be unique")
    for row in rows:
        for field in ("B_coarse_text", "C_behavior_text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"empty {field}: {row['target_id']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    targets_path = args.targets.resolve()
    out = (args.output or targets_path.parent / "embeddings").resolve()
    input_hash_before = file_sha256(targets_path)
    rows = read_targets(targets_path)

    # Imports remain local so input checks can run without loading the encoder.
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    unique_texts = list(dict.fromkeys(
        row[field] for row in rows for field in ("B_coarse_text", "C_behavior_text")
    ))
    text_index = {text: i for i, text in enumerate(unique_texts)}
    b_index = np.asarray([text_index[r["B_coarse_text"]] for r in rows], dtype=np.int64)
    c_index = np.asarray([text_index[r["C_behavior_text"]] for r in rows], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    tokenized = tokenizer(unique_texts, add_special_tokens=True, truncation=False)
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    oversized = [i for i, length in enumerate(lengths) if length > MAX_TOKENS]
    if oversized:
        raise ValueError(
            "Texts exceed 512 tokens including special tokens; refusing truncation: "
            + repr([(i, lengths[i]) for i in oversized])
        )
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
        use_safetensors=True, local_files_only=args.local_files_only,
    ).to("cpu")
    model.requires_grad_(False)
    model.eval()
    if model.config.hidden_size != DIMENSION:
        raise ValueError(f"Unexpected encoder dimension: {model.config.hidden_size}")
    if any(p.requires_grad for p in model.parameters()) or model.training:
        raise AssertionError("Encoder is not fully frozen in evaluation mode")
    chunks = []
    with torch.no_grad():
        for offset in range(0, len(unique_texts), args.batch_size):
            batch = tokenizer(
                unique_texts[offset:offset + args.batch_size], padding=True,
                truncation=False, return_tensors="pt",
            )
            cls_vectors = model(**batch).last_hidden_state[:, 0, :]
            chunks.append(F.normalize(cls_vectors, p=2, dim=1).cpu().numpy().astype(np.float32))
    vectors = np.concatenate(chunks, axis=0)
    b_vectors = vectors[b_index]
    c_vectors = vectors[c_index]
    if not np.isfinite(vectors).all():
        raise AssertionError("Non-finite text vectors")
    norms = np.linalg.norm(vectors, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 1e-5:
        raise AssertionError(f"Text vector norm error: {max_norm_error}")
    assert b_vectors.shape == c_vectors.shape == (len(rows), DIMENSION)
    # Exact reuse across B/C and duplicate target rows is a data invariant.
    for field, encoded in (("B_coarse_text", b_vectors), ("C_behavior_text", c_vectors)):
        seen = {}
        for row, vector in zip(rows, encoded):
            if row[field] in seen:
                assert np.array_equal(vector, seen[row[field]])
            seen[row[field]] = vector
    if file_sha256(targets_path) != input_hash_before:
        raise AssertionError("Input targets changed during encoding")

    out.mkdir(parents=True, exist_ok=True)
    embedding_path = out / "embeddings.npz"
    temporary_path = out / "embeddings.tmp.npz"
    np.savez_compressed(
        temporary_path,
        target_index=np.asarray([r["target_index"] for r in rows], dtype=np.int64),
        target_id=np.asarray([r["target_id"] for r in rows]),
        B_embeddings=b_vectors, C_embeddings=c_vectors,
        B_unique_text_index=b_index, C_unique_text_index=c_index,
        unique_texts=np.asarray(unique_texts), unique_embeddings=vectors,
        unique_token_lengths=np.asarray(lengths, dtype=np.int32),
    )
    os.replace(temporary_path, embedding_path)
    manifest = {
        "schema_version": "frozen-auxiliary-text-encoder-v1",
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "official_model_url": f"https://huggingface.co/{MODEL_ID}",
        "official_metadata_url": f"https://huggingface.co/api/models/{MODEL_ID}",
        "model_revision_source": "Official public Hugging Face model metadata; resolved 2026-10-15",
        "trust_remote_code": False, "weights_format": "safetensors",
        "device": "cpu", "dtype": "float32", "dimension": DIMENSION,
        "pooling": "last_hidden_state[:, 0, :] (CLS)",
        "normalization": "L2, p=2, dim=1", "text_prefix": "",
        "text_preprocessing": "exact original B/C strings; no additional normalization",
        "max_tokens_including_special_tokens": MAX_TOKENS,
        "truncation": False, "overflow_policy": "raise error before encoding",
        "eval_mode": True, "gradient_calculation": False,
        "encoder_trainable_parameters": 0, "training_runs": 0, "paid_api_calls": 0,
        "input_targets": str(targets_path), "input_sha256": input_hash_before,
        "input_unchanged_after_encoding": True,
        "target_rows": len(rows), "target_index_alignment": "targets.jsonl file order, 0..N-1",
        "target_order_sha256": value_sha256([r["target_id"] for r in rows]),
        "B_texts_sha256": value_sha256([r["B_coarse_text"] for r in rows]),
        "C_texts_sha256": value_sha256([r["C_behavior_text"] for r in rows]),
        "unique_texts": len(unique_texts),
        "unique_B_texts": len({r["B_coarse_text"] for r in rows}),
        "unique_C_texts": len({r["C_behavior_text"] for r in rows}),
        "unique_text_sha256": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in unique_texts],
        "observed_token_length_min": min(lengths), "observed_token_length_max": max(lengths),
        "max_l2_norm_error": max_norm_error,
        "all_values_finite": True, "duplicate_text_vectors_identical": True,
        "embeddings_sha256": file_sha256(embedding_path),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "library_versions": library_versions(),
        "feature_boundary": "Text vectors are training auxiliary targets only; notices are not inference features.",
    }
    (out / "encoder_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    explanation = f"""# 冻结文本编码说明

本轮把已有B粗类型说明和C具体行为说明转换为数值辅助目标，没有训练编码器、调用LLM或修改原标签。

## 固定配置

- 模型：[{MODEL_ID}](https://huggingface.co/{MODEL_ID})，官方公开中文文本编码器。
- 固定版本：`{MODEL_REVISION}`；首次下载进入用户Hugging Face缓存，后续可以离线复用。
- 使用512维CLS向量并进行L2归一化；不添加检索前缀，不截断文本。
- 全部参数冻结，使用CPU、eval模式和no_grad；B/C共用同一模型、同一维数。
- 输入：上级目录targets.jsonl；输出：本目录embeddings.npz和encoder_manifest.json。

## 本次结果与行对齐

{len(rows)}条配置目标，对应B矩阵和C矩阵各`{len(rows)}×{DIMENSION}`。相同企业年度在不同年度/时间协议中的重复配置仍保留原索引，不解释为独立样本。

两种文本合并去重后实际编码{len(unique_texts)}种文本（B有{manifest['unique_B_texts']}种，C有{manifest['unique_C_texts']}种）。含特殊标记的长度范围为{min(lengths)}—{max(lengths)} tokens，均不超过512。向量均为有限值，单位范数最大误差为{max_norm_error:.3g}；重复文本精确复用同一向量。输入文件SHA256在编码前后相同。

`B_embeddings[i]`、`C_embeddings[i]`与`targets.jsonl`中的`target_index=i`严格对应。NPZ同时保留target_id、去重文本索引与唯一文本向量。后续以每个年度/协议的辅助掩码和target_index访问；掩码关闭、索引为-1的样本不参与辅助损失，不得直接用-1索引向量。

## 训练含义

冻结文本向量用于训练期的辅助监督，帮助比较粗类型说明与具体行为说明的作用；预测时只输入原数值节点数据。当前步骤只验证编码与数据接口，模型收益留给预先固定的对照实验检验。

所有模型版本、输入与输出哈希、实际依赖版本和检查结果见encoder_manifest.json。已有向量可直接读取，不必重复下载模型。重新编码时使用项目虚拟环境运行`scripts/encode_supervision_texts.py`；模型已缓存时可加`--local-files-only`。
"""
    (out / "编码说明.md").write_text(explanation, encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "model_id", "revision", "target_rows", "unique_texts", "dimension",
        "observed_token_length_max", "max_l2_norm_error", "input_unchanged_after_encoding",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
