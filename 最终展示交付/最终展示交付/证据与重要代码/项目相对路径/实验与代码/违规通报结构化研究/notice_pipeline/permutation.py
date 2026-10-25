"""按公司/文书连接成块，构造保留年份和粗类型的细行为置换诊断。"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict

PERMUTATION_VERSION = 'case-block-permutation-v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def prepare_case_permutation(targets, provenance, seed=20261015):
    """返回与targets原顺序一致的映射；不改变原目标，不判定已可用于训练。

    每个(评价年度,时间协议)内，同公司或同候选文书的目标取传递闭包。
    仅在(经济年度,粗类型文本)多重集相同的独立块之间确定性轮换。
    """
    by_id = {}
    partitions = defaultdict(list)
    seen_keys = set()
    for position, target in enumerate(targets):
        tid = target['target_id']
        if tid in by_id:
            raise ValueError('target_id重复')
        if target['fold'] != 0:
            raise ValueError('置换仅接受训练折目标')
        if not isinstance(target['symbol'], str) or not target['symbol']:
            raise ValueError('证券代码必须为非空字符，保留前导零')
        if not target['B_coarse_text'] or not target['C_behavior_text']:
            raise ValueError('B/C文本不得为空')
        part = (int(target['evaluation_year']), target['protocol'])
        key = part + (target['symbol'], int(target['economic_year']))
        if key in seen_keys:
            raise ValueError('同一协议内公司年度重复')
        seen_keys.add(key)
        by_id[tid] = (position, target)
        partitions[part].append(tid)

    documents = defaultdict(set)
    for row in provenance:
        tid = row['target_id']
        if tid not in by_id:
            raise ValueError('来源表含未知target_id')
        if row.get('document_group'):
            documents[tid].add(row['document_group'])
    if any(not documents[tid] for tid in by_id):
        raise ValueError('部分目标缺少document_group，不能进行严格案件隔离')

    mapped = {}
    all_blocks, all_signatures, summaries = [], [], []
    for part in sorted(partitions):
        ids = sorted(partitions[part])
        parent = {tid: tid for tid in ids}

        def find(tid):
            while tid != parent[tid]:
                parent[tid] = parent[parent[tid]]
                tid = parent[tid]
            return tid

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[max(a, b)] = min(a, b)

        owners = {}
        for tid in ids:
            target = by_id[tid][1]
            keys = [('company', target['symbol'])]
            keys += [('document', d) for d in sorted(documents[tid])]
            for key in keys:
                if key in owners:
                    union(tid, owners[key])
                else:
                    owners[key] = tid
        components = defaultdict(list)
        for tid in ids:
            components[find(tid)].append(tid)

        blocks = {}
        strata = defaultdict(list)
        for tids in components.values():
            tids = sorted(tids, key=lambda tid: (
                int(by_id[tid][1]['economic_year']), by_id[tid][1]['B_coarse_text'],
                by_id[tid][1]['symbol'], tid))
            signature = tuple((int(by_id[tid][1]['economic_year']),
                               by_id[tid][1]['B_coarse_text']) for tid in tids)
            block_id = 'case-' + _digest(sorted(tids))[:20]
            blocks[block_id] = {
                'target_ids': tids, 'signature': signature,
                'companies': {by_id[tid][1]['symbol'] for tid in tids},
                'documents': set().union(*(documents[tid] for tid in tids)),
            }
            strata[signature].append(block_id)

        for signature in sorted(strata):
            block_ids = sorted(strata[signature])
            signature_id = 'signature-' + _digest(signature)[:20]
            # seed只控制块的轮换顺序；局部RNG不改变外部训练随机状态。
            shuffled = block_ids.copy()
            rng = random.Random(int(_digest([seed, part, signature])[:16], 16))
            rng.shuffle(shuffled)
            donor = {bid: shuffled[(i + 1) % len(shuffled)]
                     for i, bid in enumerate(shuffled)}
            can_exchange = len(block_ids) >= 2
            changed_count = 0
            for bid in block_ids:
                block = blocks[bid]
                source_bid = donor[bid]
                source_block = blocks[source_bid]
                if can_exchange and not block['companies'].isdisjoint(source_block['companies']):
                    raise AssertionError('跨块公司交叉，案件闭包不完整')
                all_blocks.append({
                    'evaluation_year': part[0], 'protocol': part[1],
                    'block_id': bid, 'signature_id': signature_id,
                    'target_ids': block['target_ids'],
                    'companies': sorted(block['companies']),
                    'document_groups': sorted(block['documents']),
                    'source_block_id': source_bid,
                    'can_exchange': can_exchange,
                })
                for tid, source_tid in zip(block['target_ids'], source_block['target_ids'], strict=True):
                    target, source = by_id[tid][1], by_id[source_tid][1]
                    assert (target['economic_year'], target['B_coarse_text']) == (
                        source['economic_year'], source['B_coarse_text'])
                    changed = target['C_behavior_text'] != source['C_behavior_text']
                    assert not changed or can_exchange
                    changed_count += int(changed)
                    mapped[tid] = {
                        'target_id': tid, 'target_index': target['target_index'],
                        'evaluation_year': part[0], 'protocol': part[1],
                        'symbol': target['symbol'], 'economic_year': target['economic_year'],
                        'B_coarse_text': target['B_coarse_text'],
                        'C_behavior_text': target['C_behavior_text'],
                        'D_behavior_text': source['C_behavior_text'],
                        'block_id': bid, 'signature_id': signature_id,
                        'D_source_target_id': source_tid,
                        'D_source_target_index': source['target_index'],
                        'D_source_block_id': source_bid,
                        'can_exchange_mask': can_exchange,
                        'changed_text_mask': changed,
                        'reason': ('changed' if changed else 'same_text_after_exchange'
                                   if can_exchange else 'no_matching_independent_block'),
                    }
            rows = [tid for bid in block_ids for tid in blocks[bid]['target_ids']]
            all_signatures.append({
                'evaluation_year': part[0], 'protocol': part[1],
                'signature_id': signature_id, 'signature': list(signature),
                'independent_candidate_blocks': len(block_ids), 'target_rows': len(rows),
                'can_exchange_rows': len(rows) if can_exchange else 0,
                'changed_text_rows': changed_count,
                'distinct_C_texts': len({by_id[tid][1]['C_behavior_text'] for tid in rows}),
            })

        part_rows = [mapped[tid] for tid in ids]
        assert sorted(r['D_source_target_id'] for r in part_rows) == ids
        count = len(ids)
        exchange_count = sum(r['can_exchange_mask'] for r in part_rows)
        changed_count = sum(r['changed_text_mask'] for r in part_rows)
        summaries.append({
            'evaluation_year': part[0], 'protocol': part[1], 'target_rows': count,
            'candidate_case_blocks': len(blocks), 'signatures': len(strata),
            'can_exchange_rows': exchange_count, 'changed_text_rows': changed_count,
            'exchangeable_ratio': exchange_count / count if count else 0.0,
            'changed_text_ratio': changed_count / count if count else 0.0,
            'diagnosis': ('no_exchangeable_blocks' if not exchange_count else
                          'exchangeable_but_no_text_change' if not changed_count else
                          'changed_support_requires_training_admission_review'),
        })

    ordered = [mapped[t['target_id']] for t in targets]
    total = len(ordered)
    return {
        'version': PERMUTATION_VERSION, 'seed': seed, 'mappings': ordered,
        'blocks': all_blocks, 'signature_support': all_signatures,
        'partition_summary': summaries,
        'summary': {
            'configuration_target_rows': total,
            'unique_company_years': len({(t['symbol'], t['economic_year']) for t in targets}),
            'can_exchange_rows': sum(r['can_exchange_mask'] for r in ordered),
            'changed_text_rows': sum(r['changed_text_mask'] for r in ordered),
            'changed_text_ratio': sum(r['changed_text_mask'] for r in ordered) / total if total else 0.0,
            'training_admission': 'diagnostic_only_not_an_approved_D_training_control',
        },
    }
