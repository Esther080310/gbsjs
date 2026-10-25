"""生成案件分层置换的映射和准入诊断；不编码文本、不训练、不调用API。"""
from pathlib import Path
import argparse
import csv
import json
import sys

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
import numpy as np
from notice_pipeline.permutation import prepare_case_permutation
from notice_pipeline.storage import file_digest, write_json


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=P/'artifacts/followup/supervision_targets')
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    directory = args.directory.resolve()
    files = [directory/'targets.jsonl', directory/'provenance.csv']
    before = {str(path): file_digest(path) for path in files}
    targets = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    with files[1].open(encoding='utf-8-sig', newline='') as stream:
        provenance = list(csv.DictReader(stream))
    result = prepare_case_permutation(targets, provenance, seed=args.seed)
    output = directory/'permutation_audit'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'mapping.csv', result['mappings'])
    write_csv(output/'case_blocks.csv', result['blocks'])
    write_csv(output/'signature_support.csv', result['signature_support'])
    write_csv(output/'partition_summary.csv', result['partition_summary'])
    rows = result['mappings']
    # 这组数组以targets.jsonl的原顺序保存，不能直接当作完整训练缓存的行掩码。
    np.savez_compressed(
        output/'masks.npz',
        target_id=np.asarray([r['target_id'] for r in rows], dtype=str),
        target_index=np.asarray([r['target_index'] for r in rows], dtype=np.int64),
        D_source_target_index=np.asarray([r['D_source_target_index'] for r in rows], dtype=np.int64),
        can_exchange_mask=np.asarray([r['can_exchange_mask'] for r in rows], dtype=bool),
        changed_text_mask=np.asarray([r['changed_text_mask'] for r in rows], dtype=bool),
    )
    after = {str(path): file_digest(path) for path in files}
    assert before == after, '原目标或来源表发生变化'
    summary = dict(version=result['version'], seed=args.seed,
                   **result['summary'], partitions=result['partition_summary'],
                   source_sha256=before, source_files_unchanged=True,
                   array_order='targets.jsonl original order; not full training sample order',
                   rule='connected by company or candidate document; identical multiset of (economic_year,B); block rotation')
    write_json(output/'summary.json', summary)
    lines = [
        '# D跨案置换准入诊断', '',
        '本工具检查当前目标能否构成严格的跨案细行为对照，不运行模型。', '',
        f"当前有{summary['configuration_target_rows']}条年度×时间协议配置记录，"
        f"涉及{summary['unique_company_years']}个独立企业年度。配置记录不能相加作为独立样本。", '',
        '|评价年度|时间协议|目标行|候选案件块|可交换行|实际文本改变行|改变比例|',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in result['partition_summary']:
        lines.append(f"|{row['evaluation_year']}|{row['protocol']}|{row['target_rows']}|"
                     f"{row['candidate_case_blocks']}|{row['can_exchange_rows']}|"
                     f"{row['changed_text_rows']}|{row['changed_text_ratio']:.2%}|")
    lines += ['', '规则：每个年度协议内，同公司或同候选文书连接为一个块；只轮换年份和粗类型多重集完全相同、公司集合互不相交的块。整块移动，同一块不向自身提供目标。没有同签名独立块时D保留原C并标记未改变。', '',
              '可交换只表示来源发生跨块轮换；只有C与D字符串实际不同才计入实际文本改变。当前文书键是候选分组，不能直接称为已人工确认的独立案件。', '']
    if summary['can_exchange_rows'] == 0:
        lines += ['**当前没有满足规则的可交换目标，D对照尚未准入。** 当前映射为原C的恒等映射，不能拿它训练后与C相等来评价行为监督。下一轮扩批后原规则重跑；不放宽年份、粗类型或案件隔离条件凑置换。', '']
    elif summary['changed_text_rows'] == 0:
        lines += ['**当前跨块轮换没有产生细行为文本差别，D对照尚未准入。** 扩批后需要检查同粗类型内部的不同细行为支持量。', '']
    else:
        lines += ['当前存在发生文本改变的跨块映射；仍需根据各年度协议的支持量及共同训练掩码决定D训练准入。本工具不自动批准训练。', '']
    lines += ['文件：mapping.csv保存逐目标来源和C/D文本；case_blocks.csv保存连通块；signature_support.csv保存每个签名的独立块与细行为支持量；partition_summary.csv/summary.json保存计数；masks.npz保存可交换、实际改变两个不同掩码。', '',
              'masks.npz与targets.jsonl逐行同序，不是完整训练缓存掩码；后续读取须按target_index和原样本键再对齐。原C、原辅助掩码、原Y和模型均未修改；没有API调用。', '']
    (output/'准入诊断.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
