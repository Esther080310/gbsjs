"""输出与冻结缓存逐行对齐的辅助监督文本、索引和掩码，不训练或调用API。"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0,str(P))
import numpy as np
from notice_pipeline.supervision import SUPERVISION_VERSION, behavioral_target, merge_targets, PROPOSITIONS, OBJECTS

ROOT = P/'artifacts/followup'
OUT = ROOT/'supervision_targets'
TYPE_NAMES = {'P2501':'虚构利润','P2503':'虚假记载或误导性陈述','P2504':'推迟披露',
              'P2505':'重大遗漏','P2506':'其他披露不实','P2509':'擅自改变资金用途',
              'P2510':'占用公司资产','P2514':'违规担保','P2515':'一般会计处理不当','P2599':'其他违规'}


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_csv(name, rows, fields):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'fact_roles.csv',ROOT/'event_sample_links.csv',
           E/'财务年度配对修正/cache/2020.npz',E/'财务年度配对修正/cache/2021.npz']
    source_summary_file=ROOT/'offline_summary.json'
    if source_summary_file.exists(): files.append(source_summary_file)
    before={str(p):sha(p) for p in files}
    facts={(r['record_id'],r['fact_id']):r for r in read_csv(files[0])}
    links=read_csv(files[1])
    source_summary=json.loads(source_summary_file.read_text()) if source_summary_file.exists() else {}
    # 直接读取源码常量，避免导入训练模块时加载Torch并改变随机状态。
    tree=ast.parse((E/'发生年度实验/common.py').read_text())
    codes=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='TYPE_CODES' for t in n.targets))
    assert set(codes)==set(TYPE_NAMES)
    audits=[]
    templates={}
    for key,fact in facts.items():
        reasons=[]
        if fact['fact_kind']!='violation_behavior': reasons.append('非独立违规行为')
        if fact['target_relation']!='target_company': reasons.append('非目标公司主体或不明')
        if fact.get('pilot_split')=='holdout': reasons.append('原试点保留组不参与辅助训练')
        evidence=json.loads(fact['evidence'])
        if not any(e['field']=='Activity' for e in evidence): reasons.append('缺少Activity行为证据')
        item=behavioral_target(fact['behavior'])
        if not item['text']: reasons.append('无法形成具体行为短语')
        templates[key]=dict(item,eligible=not reasons)
        audits.append(dict(record_id=key[0],fact_id=key[1],source_status=fact['source_status'],
            summary_text=item['text'] if not reasons else '',proposition_ids=';'.join(item['proposition_ids']),
            object_ids=';'.join(item['object_ids']),eligible_semantics=not reasons,
            exclusion_reason=';'.join(reasons)))
    write_csv('fact_supervision_audit.csv',audits,list(audits[0]))
    target_rows=[]
    coverage=[]
    provenance=[]
    for protocol in ['retrospective_snapshot2023','asof_training_year_end']:
        for ey in (2020,2021):
            groups=defaultdict(list)
            for link in links:
                if int(link['evaluation_year'])!=ey or link['fold']!='0' or link['admissible']!='True': continue
                key=(link['record_id'],link['fact_id'])
                if not templates[key]['eligible']: continue
                if protocol=='asof_training_year_end' and link['declared_by_training_cutoff']!='True': continue
                groups[link['symbol'],int(link['economic_year'])].append(link)
            arrays={}
            with np.load(E/f'财务年度配对修正/cache/{ey}.npz',allow_pickle=False) as z:
                for fold in range(4):
                    symbols=z[f'{fold}_symbol'].astype(str)
                    years=z[f'{fold}_year']
                    n=len(years)
                    mask=np.zeros(n,dtype=bool)
                    ids=np.full(n,-1,dtype=np.int64)
                    case_counts=np.zeros(n,dtype=np.int32)
                    for i,(symbol,year) in enumerate(zip(symbols,years)):
                        relevant=groups.get((symbol,int(year)),[]) if fold==0 else []
                        if not relevant: continue
                        assert z[f'{fold}_y'][i]==1
                        merged=merge_targets([templates[r['record_id'],r['fact_id']] for r in relevant])
                        coarse=[TYPE_NAMES[c] for c,flag in zip(codes,z[f'{fold}_types'][i]) if flag>0]
                        if not coarse: continue  # B/C使用相同可用样本；不编造十头外粗类型。
                        ti=len(target_rows)
                        target_id=f'{protocol}:{ey}:0:{i}'
                        target_rows.append(dict(target_index=ti,target_id=target_id,protocol=protocol,
                            evaluation_year=ey,fold=fold,row_index=i,symbol=symbol,economic_year=int(year),
                            B_coarse_text='违规类型：'+'；'.join(coarse)+'。',C_behavior_text=merged['text'],
                            unique_behavior_summaries=merged['unique_behavior_summaries'],
                            source_fact_count=len({(r['record_id'],r['fact_id']) for r in relevant}),
                            document_groups=len({r['document_group'] for r in relevant})))
                        mask[i]=True;ids[i]=ti
                        case_counts[i]=len({r['document_group'] for r in relevant})
                        for link in relevant:
                            fact=facts[link['record_id'],link['fact_id']]
                            provenance.append(dict(target_id=target_id,record_id=link['record_id'],fact_id=link['fact_id'],
                                source_status=fact['source_status'],document_group=link['document_group'],
                                original_behavior=fact['behavior'],evidence=fact['evidence'],
                                original_time_literal=fact['time_literal'],time_role=fact['time_role'],
                                declare_date=fact['declare_date']))
                    arrays.update({f'{fold}_symbol':symbols,f'{fold}_year':years,
                        f'{fold}_aux_mask':mask,f'{fold}_target_index':ids,f'{fold}_source_case_count':case_counts})
                    coverage.append(dict(protocol=protocol,evaluation_year=ey,fold=fold,
                        samples=n,aux_samples=int(mask.sum()),aux_fraction=float(mask.mean()),
                        auxiliary_weight_rule='每个企业年度等权；无目标mask=0，原主损失照常计算'))
                    assert np.all(ids[~mask]==-1)
                    if fold!=0: assert not mask.any()
            np.savez_compressed(OUT/f'{ey}_{protocol}.npz',**arrays)
    with (OUT/'targets.jsonl').open('w',encoding='utf-8') as f:
        for row in target_rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_csv('targets_preview.csv',target_rows,list(target_rows[0]) if target_rows else ['target_id','B_coarse_text','C_behavior_text'])
    write_csv('coverage.csv',coverage,list(coverage[0]))
    write_csv('provenance.csv',provenance,list(provenance[0]) if provenance else ['target_id','record_id','fact_id'])
    # 不生成只有正例的BCE标签；文本目标由同一个冻结编码器向量化。
    manifest=dict(version=SUPERVISION_VERSION,input_sha256=before,type_codes=list(codes),type_names=TYPE_NAMES,
        target_format='组合式行为文本；由LLM事实behavior中的明确短语生成，非开放式再次摘要',
        vocabulary=dict(propositions=PROPOSITIONS,objects=OBJECTS),
        target_rows=len(target_rows),fact_count=len(facts),source_processed_records=source_summary.get('processed_records'),
        source_pending_records=source_summary.get('pending_records'),
        unique_company_years=len({(r['symbol'],r['economic_year']) for r in target_rows}),
        unique_behavior_texts=len({r['C_behavior_text'] for r in target_rows}),
        used_unique_facts=len({(r['record_id'],r['fact_id']) for r in provenance}),
        used_document_groups=len({r['document_group'] for r in provenance}),
        semantic_eligible_facts=sum(t['eligible'] for t in templates.values()),
        coverage=coverage,source_status_counts=dict(Counter(r['source_status'] for r in audits)),
        raw_y_modified=False,api_calls=0,training_runs=0,embedding_encoder=None,
        downstream_rule='B/C使用同一冻结编码器、同维目标、同mask；主模型输入不读取通报；未知目标mask=0',
        pending='当前目标需用已固定的冻结编码器重新编码；继续已获授权的首批抽取，并重查D置换支持量')
    encoder_file=OUT/'embeddings/encoder_manifest.json'
    if encoder_file.exists():
        encoder=json.loads(encoder_file.read_text())
        if encoder['input_sha256']==sha(OUT/'targets.jsonl'):
            manifest['embedding_encoder']={k:encoder[k] for k in ('model_id','revision','dimension','pooling','normalization')}
            manifest['pending']='继续已获授权的首批抽取；重建目标并检查D置换支持量，再补齐统一训练驱动'
    assert before=={str(p):sha(p) for p in files},'输入发生改变'
    with (OUT/'manifest.json').open('w',encoding='utf-8') as f: json.dump(manifest,f,ensure_ascii=False,indent=2)
    lines=['# 辅助监督数据处理结果','',
        f"版本：{SUPERVISION_VERSION}。本次离线整理复用{source_summary.get('processed_records','已有')}条已处理通报；源run尚有{source_summary.get('pending_records','未知')}条待处理。本脚本不调用LLM、不训练模型。",'',
        f"{len(facts)}项保留事实中，{manifest['semantic_eligible_facts']}项同时满足行为/目标公司/正文证据/可生成具体短语的文本条件。再经过年度、原Y、样本与案件跨折限制，最终采用{manifest['used_unique_facts']}项不同事实，涉及{manifest['used_document_groups']}个候选文书组、{manifest['unique_company_years']}个独立企业年度、{manifest['unique_behavior_texts']}种不同的合并说明。",'',
        f"两套年度×两套时间协议共生成{len(target_rows)}条配置记录，包含相同企业年度的重复使用，不是{len(target_rows)}个独立样本。原试点保留文书组已从辅助训练中隔离。",'',
        '|时间协议|评价年度|训练样本|有辅助目标的企业年度|覆盖率|',
        '|---|---:|---:|---:|---:|']
    for r in coverage:
        if r['fold']==0:
            label='2023快照回顾性' if r['protocol']=='retrospective_snapshot2023' else '训练年度末前已公开'
            lines.append(f"|{label}|{r['evaluation_year']}|{r['samples']}|{r['aux_samples']}|{r['aux_fraction']:.4%}|")
    lines += ['',
        '“训练年度末前已公开”分别按2017-12-31/2018-12-31筛通报；这是保守时间协议，不把它当作原模型实际拟合时间。回顾性协议沿用2023标签快照，允许事后披露的训练样本行为用于回顾性方法检验。两者分别保存。', '',
        '本轮产物：targets.jsonl提供B原类型说明、C细行为说明；四份NPZ提供与冻结缓存逐行相同的样本键、目标索引和掩码；provenance.csv提供原行为、事实编号及证据；fact_supervision_audit.csv提供全部事实的文本准入去向；coverage.csv及manifest.json给出覆盖和版本。', '',
        '目标文本由18种行为短语、9种事项短语组合，匹配LLM已抽取的单项behavior。公司名、人员名、具体日期、金额及处分不进入目标文本；原值保存在来源表。组合规则没有匹配的事实弃权，未把未知改为负例。当前方法是规则规范化后的LLM事实监督，不等同于保留了原自由文本的全部语义。', '',
        '原Y和十个类型任务继续使用原缓存。只有训练折能打开辅助掩码；选模、校准、年度评价折全部关闭。B/C使用相同掩码和样本，供后续区分额外辅助训练与细行为内容的作用。', '',
        (f"已用冻结BAAI/bge-small-zh-v1.5编码当前B/C目标，各{len(target_rows)}×512维。具体commit和验证见embeddings/encoder_manifest.json；训练接口返回目标库、样本索引和掩码，来源证据保留在独立表中。" if manifest['embedding_encoder'] else
         '当前已固定文本、来源与样本配对；下一步用固定的BAAI/bge-small-zh-v1.5重新编码目标。'), '',
        '本轮辅助覆盖很少，多数普通小批次会没有辅助目标。现阶段用于验证数据流和构造性，正式效果比较先定向补充训练期通报。']
    (OUT/'处理结果.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:manifest[k] for k in ['version','fact_count','semantic_eligible_facts','used_unique_facts',
        'used_document_groups','unique_company_years','unique_behavior_texts','target_rows']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
