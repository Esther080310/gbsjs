"""离线整理既有抽取；规则只生成候选，不修改Y或冻结缓存，不调用API。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(P))
import numpy as np
from notice_pipeline.source import load_config, run_path, verify_run, date_upper_bound
from notice_pipeline.pipeline import valid_results
from notice_pipeline.recovery import partial_results
from notice_pipeline.storage import read_jsonl

OUT = P / 'artifacts/followup'
RULES = {
    'fabricated_trade': {'name': '虚构交易或业务', 'pattern': r'虚构|虚假交易|伪造|捏造|无真实.*(?:业务|交易)|虚增(?:收入|利润)'},
    'recognition_timing': {'name': '确认时点', 'pattern': r'提前确认|延[迟后]确认|跨期|收入确认时点'},
    'measurement_impairment': {'name': '计量及减值', 'pattern': r'减值|坏账|计提|公允价值|会计估计'},
    'undisclosed_related_funds': {'name': '关联资金占用或隐瞒', 'pattern': r'资金占用|占用.*资金|未.*(?:记账|入账)|非经营性.*(?:资金|往来)'},
    'approval_disclosure_gap': {'name': '审批与披露脱节', 'pattern': r'未经.*(?:审议|审批)|未履行.*(?:审议|审批)|超.*(?:审批|批准|授权)|未.*披露.*担保|违规担保'},
    'disclosure_gap_other': {'name': '其他披露缺口', 'pattern': r'未.*披露|披露.*(?:不实|不符)|不准确|虚假记载|重大遗漏|误导性'},
    'securities_rule': {'name': '证券交易规则事项', 'pattern': r'短线交易|敏感期|窗口期|违规.*(?:买卖|减持|增持)'},
    'governance_duty': {'name': '治理职责与期限', 'pattern': r'空缺超过|未.*代行|违规.*任职|未.*勤勉|未.*尽职'},
}
WRONG = r'违反|违规|虚构|伪造|虚增|虚减|未按|未及时|未履行|未披露|未予披露|未将|不符|不实|不准确|遗漏|误导|非经营性占用|资金占用|提前确认|少计|多计|不符合|隐瞒|挪用|擅自|未经|未能|未尽|超出|超过.*(?:额度|期限|授权)'
SANCTION = r'罚款|警告|禁入|谴责|通报批评|监管谈话|警示函|责令|整改|严肃批评|予以.*处分|没收'


def dump_csv(name, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def symbol(value):
    s = str(value).strip()
    return s.zfill(6) if re.fullmatch(r'\d{1,6}', s) else ''


def pairing_years(time):
    if time.get('status') != 'explicit' or time.get('role') not in {'occurrence','reporting_period'}:
        return []
    return time.get('years') or []


def classify(fact, source):
    """事实行为优先；不以LLM机制枚举或整段案件语义传播细标签。"""
    b = fact['behavior']
    fields = {q['field'] for q in fact['evidence']}
    has_wrong = bool(re.search(WRONG, b))
    punishment = bool(re.search(SANCTION, b))
    if fields == {'PunishmentMeasure'} and punishment:
        kind, basis = 'sanction', '仅处分字段且行为描述为处分措施'
    elif punishment and has_wrong:
        kind, basis = 'mixed', '同一行为描述同时含处分与违规表述'
    elif punishment:
        kind, basis = 'sanction', '行为描述为处分措施'
    elif has_wrong:
        kind, basis = 'violation_behavior', '行为描述有可定位的不当行为线索；规则候选'
    elif re.search(r'实际控制人.*变更|变更.*实际控制人|签订|成立|收到|支付|归还|担任|离职|起诉', b):
        kind, basis = 'background', '未独立表述违规的背景或交易步骤；保留供案件解释'
    else:
        kind, basis = 'unknown', '行为描述不足以独立确认角色'
    a = fact.get('actor', {})
    name, role = a.get('name') or '', a.get('role')
    names = [source.get('CoFullName', ''), source.get('ShortName', '')]
    match = name and any(name in n or n in name for n in names if n)
    if role in {'executive','controlling_shareholder','other_shareholder','auditor','intermediary','subsidiary','related_party'}:
        target = 'individual_or_other'
    elif role == 'listed_company' and (match or name in {'公司','本公司','你公司','该公司'}):
        target = 'target_company'
    else:
        target = 'unclear'
    labels = [k for k, v in RULES.items() if re.search(v['pattern'], b)] if kind == 'violation_behavior' else []
    return kind, basis, target, labels


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_runs(config_paths):
    """合并同快照run；同源记录去重、完整结果优先，原pilot保留组不变。"""
    records_by_source, rid_sources, selected = {}, {}, {}
    source_paths, run_summaries, seen_runs = [], [], set()
    pilot_splits, pilot_holdout_groups = {}, set()
    source_sha256 = None
    for run_number, config_path in enumerate(config_paths):
        config_path = Path(config_path).resolve()
        cfg = load_config(config_path)
        if cfg['study']['snapshot_cutoff'] != '2023-12-31':
            raise ValueError('所有源run必须使用2023-12-31标签快照')
        run_manifest = verify_run(cfg)
        run = run_path(cfg).resolve()
        if run in seen_runs:
            continue
        seen_runs.add(run)
        if run_manifest['config']['study']['snapshot_cutoff'] != '2023-12-31':
            raise ValueError('源run manifest快照与当前协议不一致')
        if source_sha256 is None:
            source_sha256 = run_manifest['source_sha256']
        elif source_sha256 != run_manifest['source_sha256']:
            raise ValueError('源run使用了不同原Y版本，不能直接合并')
        prepared = read_jsonl(run/'prepared.jsonl')
        completed = valid_results(cfg, prepared)
        partials = {rid: row for rid, row in partial_results(cfg, prepared).items()
                    if rid not in completed}
        source_paths += [config_path, run/'manifest.json']
        source_paths += [run/name for name in ('prepared.jsonl', 'extractions.jsonl', 'partials.jsonl')
                         if (run/name).exists()]
        run_id = cfg['study']['run_id']
        run_summaries.append(dict(run_id=run_id, config_path=str(config_path), run_path=str(run),
            snapshot_cutoff=cfg['study']['snapshot_cutoff'], prepared_records=len(prepared),
            complete_records=len(completed), partial_records=len(partials),
            processed_records=len(completed)+len(partials),
            pending_records=len(prepared)-len(completed)-len(partials)))
        for record in prepared:
            rid = record['record_id']
            identity = hashlib.sha256(json.dumps(record['source'], ensure_ascii=False,
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if rid in rid_sources and rid_sources[rid] != identity:
                raise ValueError(f'同record_id的原始source不一致：{rid}')
            rid_sources[rid] = identity
            if run_number == 0:
                pilot_splits[identity] = record['split']
                if record['split'] == 'holdout':
                    pilot_holdout_groups.add(record['document_group'])
            if identity not in records_by_source:
                records_by_source[identity] = dict(record, _source_runs=[])
            canonical = records_by_source[identity]
            if canonical['document_group'] != record['document_group']:
                raise ValueError(f'同源记录候选文书组发生变化：{rid}')
            origin = dict(run_id=run_id, run_path=str(run), record_id=rid, split=record['split'])
            canonical['_source_runs'].append(origin)
            status = 'complete' if rid in completed else 'partial' if rid in partials else None
            if status is not None:
                existing = selected.get(identity)
                if existing is None or (existing[0] == 'partial' and status == 'complete'):
                    selected[identity] = (status, (completed if status == 'complete' else partials)[rid], origin)
    records, complete, partial = [], {}, {}
    for identity, record in records_by_source.items():
        record['_pilot_split'] = pilot_splits.get(identity, '')
        record['_pilot_holdout_group'] = (record['document_group'] in pilot_holdout_groups
                                          or record['_pilot_split'] == 'holdout')
        if identity in selected:
            status, result, origin = selected[identity]
            record['_selected_source'] = origin
            (complete if status == 'complete' else partial)[record['record_id']] = result
        records.append(record)
    return records, complete, partial, run_summaries, source_paths, pilot_holdout_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--additional-config', action='append', type=Path, default=[],
                        help='可重复：合并额外已prepare的同快照配置，默认始终保留pilot')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records, complete, partial, run_summaries, source_paths, pilot_holdout_groups = collect_runs(
        [P/'configs/pilot.toml', *args.additional_config])
    result = {**partial, **complete}
    protected = [E / '财务年度配对修正/cache' / f'{y}.npz' for y in (2020,2021)]
    manifest = json.loads((E / '最终版本冻结/final_manifest.json').read_text())
    assert manifest['label_cutoff'] == '2023-12-31'
    protected += source_paths + [E / '大创数据集/Y 经营违规公司名录/STK_Violation_Main.xlsx']
    before = {str(p):fingerprint(p) for p in protected}
    indices = {}
    denominators = {}
    for ey in (2020,2021):
        index = {}
        with np.load(E / f'财务年度配对修正/cache/{ey}.npz', allow_pickle=False) as z:
            for fold in range(4):
                ss, yy, labels = z[f'{fold}_symbol'], z[f'{fold}_year'], z[f'{fold}_y']
                denominators[ey,fold] = len(ss)
                for s, y, label in zip(ss, yy, labels):
                    key = (symbol(s), int(y))
                    assert key not in index, (ey,key)
                    index[key] = (fold,int(label))
        indices[ey] = index
    facts, links, conflicts = [], [], []
    groups = defaultdict(set)
    for record in records:
        rid, source = record['record_id'], record['source']
        if rid not in result:
            continue  # 新run尚未调用/尚未形成可保留结果：只计pending，不生成事实。
        origin = record['_selected_source']
        provenance = dict(source_run_id=origin['run_id'], source_run_path=origin['run_path'],
            source_record_id=origin['record_id'], source_split=origin['split'],
            source_runs=json.dumps(record['_source_runs'], ensure_ascii=False),
            pilot_split=record['_pilot_split'], pilot_holdout_group=record['_pilot_holdout_group'])
        source_years = set(map(int, re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', source['ViolationYear'])))
        pairs = list(zip(source['ViolationYear'].split(';'), source['ViolationTypeID'].split(';')))
        for fact in result[rid]['extraction']['facts']:
            kind,basis,target,labels = classify(fact,source)
            time = fact['time']
            years = time.get('years') or []
            role_ok = time['role'] in {'occurrence','reporting_period'}
            time_ok = bool(pairing_years(time))
            mismatch = bool(years and source_years and set(years)-source_years and role_ok)
            base = dict(record_id=rid, fact_id=fact['fact_id'], source_status='complete' if rid in complete else 'partial',
                        document_group=record['document_group'], **provenance,
                        symbol=symbol(source['Symbol']), company=source['CoFullName'],
                        original_is_violated=source['IsViolated'], original_year_type_pairs=json.dumps(pairs,ensure_ascii=False),
                        fact_kind=kind,target_relation=target,actor=json.dumps(fact['actor'],ensure_ascii=False),
                        behavior=fact['behavior'], candidate_behavior=';'.join(labels) or 'unknown',
                        llm_mechanism_candidate=fact['mechanism'], time_role=time['role'],
                        time_status=time['status'], time_literal=time.get('literal'),
                        time_years=json.dumps(years), declare_date=source['DeclareDate'],
                        evidence=json.dumps(fact['evidence'],ensure_ascii=False), classification_basis=basis,
                        supervision_status='candidate_only_not_frozen')
            facts.append(base)
            if mismatch:
                conflicts.append(dict(record_id=rid,fact_id=fact['fact_id'],symbol=base['symbol'],
                    source_run_id=origin['run_id'],pilot_split=record['_pilot_split'],
                    original_year=source['ViolationYear'],fact_years=json.dumps(years),time_role=time['role'],
                    behavior=fact['behavior'],reason='事实期间包含原记录年度之外的年份，保留角色差异，不覆盖Y'))
            for year in (years if time_ok else [None]):
                for ey,index in indices.items():
                    found = index.get((base['symbol'],year)) if year is not None else None
                    reasons = []
                    if kind != 'violation_behavior': reasons.append('非独立违规行为')
                    if target != 'target_company': reasons.append('主体非目标公司或不明确')
                    if source['IsViolated'] != 'Y': reasons.append('原记录非上市公司违规Y')
                    if not labels: reasons.append('尚无具体行为候选')
                    if not time_ok: reasons.append('无明确发生年或报表年')
                    if mismatch: reasons.append('期间与原年度有差异待厘清')
                    if year is not None and year not in source_years: reasons.append('未与原标签年度交集')
                    if not found: reasons.append('冻结样本无匹配')
                    elif found[1] != 1: reasons.append('冻结样本原风险Y非正例')
                    if found and found[0] == 0 and record['_pilot_holdout_group']:
                        reasons.append('原pilot保留文书组，排除训练候选')
                    row = dict(record_id=rid,fact_id=fact['fact_id'],document_group=record['document_group'],
                        **provenance,
                        source_status=base['source_status'],symbol=base['symbol'],economic_year=year,
                        evaluation_year=ey,fold=found[0] if found else '',original_sample_y=found[1] if found else '',
                        time_role=time['role'], pairing_basis='抽取事实明确期间，与原Y年度交集；无公告年填补',
                        candidate_behavior=base['candidate_behavior'],
                        preliminary_admissible=not reasons, exclusion_reason=';'.join(reasons),
                        declared_by_training_cutoff=bool(date_upper_bound(source['DeclareDate']) and
                            str(date_upper_bound(source['DeclareDate'])) <= f'{ey-3}-12-31'))
                    links.append(row)
                    if found and kind == 'violation_behavior' and target == 'target_company':
                        groups[ey,record['document_group']].add(found[0])
    cross = {k:v for k,v in groups.items() if len(v)>1}
    for row in links:
        row['case_cross_fold'] = (row['evaluation_year'],row['document_group']) in cross
        row['admissible'] = row['preliminary_admissible'] and not row['case_cross_fold']
        if row['case_cross_fold']:
            row['exclusion_reason'] += (';' if row['exclusion_reason'] else '')+'同候选文书跨折，需案件分组协议'
    assert len(facts) == len({(r['record_id'],r['fact_id']) for r in facts})
    assert len(facts) == sum(len(v['extraction']['facts']) for v in result.values())
    dump_csv('fact_roles.csv', facts)
    dump_csv('event_sample_links.csv', links)
    dump_csv('year_conflicts.csv', conflicts, ['record_id','fact_id','symbol','source_run_id','pilot_split',
        'original_year','fact_years','time_role','behavior','reason'])
    coverage = []
    for ey in (2020,2021):
        for fold in range(4):
            subset = [r for r in links if r['evaluation_year']==ey and r['fold']==fold]
            for behavior in ['ALL', *RULES]:
                matched = [r for r in subset if behavior=='ALL' or behavior in r['candidate_behavior'].split(';')]
                eligible = [r for r in matched if r['admissible']]
                keys = {(r['symbol'],r['economic_year']) for r in eligible}
                coverage.append(dict(evaluation_year=ey,fold=fold,behavior=behavior,
                    all_samples=denominators[ey,fold],matched_fact_year_links=len(matched),
                    admissible_fact_year_links=len(eligible),unique_company_years=len(keys),
                    document_groups=len({r['document_group'] for r in eligible}),
                    record_count=len({r['record_id'] for r in eligible}),
                    asof_training_eligible_links=sum(r['declared_by_training_cutoff'] for r in eligible),
                    pilot_holdout_training_links=sum(r['pilot_holdout_group'] and fold==0 for r in matched),
                    case_cross_fold_links=sum(r['case_cross_fold'] for r in matched)))
    dump_csv('supervision_coverage.csv', coverage)
    coarse = defaultdict(list)
    for fact in facts:
        if fact['fact_kind']=='violation_behavior' and fact['candidate_behavior']!='unknown':
            for _,codes in json.loads(fact['original_year_type_pairs']):
                for code in re.split('[、,，]', codes):
                    for label in fact['candidate_behavior'].split(';'):
                        if code.strip(): coarse[code.strip(),label].append(fact)
    dump_csv('coarse_type_behavior_support.csv', [dict(original_type=code,candidate_behavior=label,
        fact_count=len({(f['record_id'],f['fact_id']) for f in fs}),
        record_count=len({f['record_id'] for f in fs}),document_groups=len({f['document_group'] for f in fs}),
        meaning='记录级粗类型共现，不是逐事实类型真值') for (code,label),fs in sorted(coarse.items())])
    dump_csv('case_cross_fold.csv', [dict(evaluation_year=y,document_group=g,folds=';'.join(map(str,sorted(fs))))
             for (y,g),fs in cross.items()], ['evaluation_year','document_group','folds'])
    dictionary = dict(version='candidate-rules-v1',status='规则候选，非人工标注真值；不直接冻结训练目标',
                      rules=RULES, unknown='不满足候选规则；保留原行为和证据',
                      labels_are_multilabel=True, negative_examples='未提及不等于行为不存在')
    (OUT/'behavior_dictionary.json').write_text(json.dumps(dictionary,ensure_ascii=False,indent=2))
    summary = dict(frozen_version=manifest['version'],records=len(records),prepared_records=len(records),
        processed_records=len(result),pending_records=len(records)-len(result),
        complete_records=len(complete),partial_records=len(partial),source_runs=run_summaries,
        pilot_holdout_document_groups=len(pilot_holdout_groups),
        pilot_holdout_facts=sum(r['pilot_holdout_group'] for r in facts),
        pilot_holdout_training_links=sum(r['pilot_holdout_group'] and r['fold']==0 for r in links),
        facts=len(facts),fact_kinds=dict(Counter(r['fact_kind'] for r in facts)),
        target_relations=dict(Counter(r['target_relation'] for r in facts)),year_conflict_facts=len(conflicts),
        candidate_counts=dict(Counter(l for r in facts for l in r['candidate_behavior'].split(';'))),
        cross_fold_candidate_groups=len(cross),
        unmatched_fact_counts={str(ey):len({(r['record_id'],r['fact_id']) for r in links if r['evaluation_year']==ey and r['fold']==''}) for ey in (2020,2021)},
        protected_sha256=before,api_calls=0,training_runs=0)
    assert before == {str(p):fingerprint(p) for p in protected}, '保护输入发生变化'
    (OUT/'offline_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines = ['# 覆盖与准入小结','',f"冻结版本：{manifest['version']}。本轮合并{len(run_summaries)}个已校验run，用于监督构造与字段需求分析。",'',
        f"同源去重后已准备{len(records)}条记录，已处理{len(result)}条（{len(complete)}条完整、{len(partial)}条部分），待处理{len(records)-len(result)}条；当前共{len(facts)}项保留事实，隔离项未纳入。",'',
        f"原pilot保留{len(pilot_holdout_groups)}个候选文书组，{summary['pilot_holdout_facts']}项保留组事实继续用于审计和字段分析；训练候选排除其{summary['pilot_holdout_training_links']}条事实—年度链接。新增run的development划分不覆盖原保留用途。",'',
        '规则分类：'+json.dumps(summary['fact_kinds'],ensure_ascii=False)+'。',
        '主体关系：'+json.dumps(summary['target_relations'],ensure_ascii=False)+'。','',
        f'发现{len(conflicts)}项事实期间与原年度存在差异，{len(cross)}个评价年度—候选文书组跨折。具体见对应CSV。', '',
        '|评价年度|折|全部样本|候选准入企业年度|候选文书组|拟合截点前公开的准入事实年链接|',
        '|---|---|---:|---:|---:|---:|']
    for row in coverage:
        if row['behavior']=='ALL':
            lines.append('|'+ '|'.join(str(row[k]) for k in ['evaluation_year','fold','all_samples','unique_company_years','document_groups','asof_training_eligible_links'])+'|')
    lines += ['', '折0训练、折1选模、折2校准、折3年度评价。不同评价年度不能合并计算覆盖。', '',
        '准入是构造候选：需同时满足独立行为、上市公司主体、原公司违规Y、明确发生/报表期间、与原标签年度一致、冻结样本Y为1且无跨折案件冲突；训练候选还排除原pilot保留文书组。最终辅助目标由后续监督构造步骤另行生成。', '',
        '时间严格的部署协议还需明确拟合日期与文书首次公开日。当前公开截止比较采用训练年度末的保守检查，不能替代模型实际拟合日期。', '',
        '原始Y、样本键、缓存顺序、现有抽取内容均未修改，已核对输入前后SHA256。本脚本没有API调用或训练。', '',
        '细标签依据单项行为文字；不把整个案件的词汇传给每项事实。多个标签可同时出现，未提及不构造负例。主体不明、背景、处分和时间不明均保留供案件解释。', '',
        '下一步以案例和新观测为依据确定优先机制；如需要更细主体或语义字段，可在T4提出具体扩批，而非强行把未知项纳入训练。']
    (OUT/'覆盖与准入小结.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='protected_sha256'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
