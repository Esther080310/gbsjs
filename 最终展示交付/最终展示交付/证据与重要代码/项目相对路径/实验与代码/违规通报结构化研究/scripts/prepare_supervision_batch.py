"""准备明确的首批清单及费用情景；只生成文件，不调用DeepSeek。"""
from pathlib import Path
from collections import Counter, defaultdict
import csv,json,sys
import numpy as np
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from notice_pipeline.source import load_config,read_source,resolve,run_path,sample_groups
from notice_pipeline.storage import read_jsonl,file_digest,write_json
from notice_pipeline.pipeline import body_for

DEST=P/'artifacts/followup/supervision_setup'
INVENTORY=P/'artifacts/followup/training_notice_inventory/expansion_call_candidates.csv'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    cfg=load_config(P/'configs/pilot.toml')
    with INVENTORY.open(encoding='utf-8-sig',newline='') as f: queue={r['record_id']:r for r in csv.DictReader(f)}
    records=read_source(resolve(cfg['source']['path']))
    byid={r['record_id']:r for r in records}
    # 先用两套评价年度均可配的回顾性范围；保证第一轮可分别报告两年。
    common={rid:r for rid,r in queue.items() if all(f'{ey}:retrospective_snapshot2023' in json.loads(r['candidate_protocols']) for ey in (2020,2021))}
    selected=sample_groups([byid[rid] for rid in sorted(common)],500,cfg['study']['seed'],0)
    ids=[r['record_id'] for r in selected]
    assert set(ids).issubset(common) and len(ids)==len(set(ids))
    assert not set(ids)&{r['record_id'] for r in read_jsonl(run_path(cfg)/'prepared.jsonl')}
    for record in selected:
        assert record['source']['IsViolated'].strip().upper()=='Y'
    selection=dict(version='explicit-batch1-v1',source_sha256=file_digest(resolve(cfg['source']['path'])),
        inventory_sha256=file_digest(INVENTORY),record_ids=ids,seed=cfg['study']['seed'],
        scope='2020/2021两套协议共同训练范围；2023快照回顾性',
        selection='复用既有whole-document-group分层抽样；按原类型粗分、文本长短与年份完整性平衡，非按模型分数选择',
        prospective_training_allowed=False,api_calls=0)
    write_json(DEST/'batch1_record_ids.json',selection)
    with (DEST/'batch1_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(next(iter(queue.values()))));w.writeheader();w.writerows(queue[rid] for rid in ids)
    pilot=read_jsonl(run_path(cfg)/'prepared.jsonl')
    old_byid={r['record_id']:r for r in pilot}
    responses={}
    for filename in ['extractions.jsonl','partials.jsonl','failures.jsonl']:
        for row in read_jsonl(run_path(cfg)/filename):
            if row.get('raw_response_file') and row['record_id'] in old_byid:
                responses[row['raw_response_file']]=row['record_id']
    empirical=[]
    for filename,rid in responses.items():
        raw=json.loads((run_path(cfg)/filename).read_text())
        usage=raw['usage']
        chars=len(json.dumps(body_for(cfg,old_byid[rid]),ensure_ascii=False))
        empirical.append(dict(chars=chars,prompt=usage['prompt_tokens'],output=usage['completion_tokens'],
            hit=usage.get('prompt_cache_hit_tokens',0),miss=usage.get('prompt_cache_miss_tokens',usage['prompt_tokens'])))
    assert len(empirical)==len(list((run_path(cfg)/'raw_responses').glob('*.json')))
    ratio=float(np.median([r['prompt']/r['chars'] for r in empirical]))
    token_inputs=[max(1,round(len(json.dumps(body_for(cfg,r),ensure_ascii=False))*ratio)) for r in selected]
    # 从旧102份响应的输出分布构造费用情景；不把字符数当精确token。
    outputs=np.array([r['output'] for r in empirical])
    rates={'off_peak':{'input_miss':1.,'input_hit':.02,'output':4.},
           'peak':{'input_miss':2.,'input_hit':.04,'output':8.}}
    cost=[]
    for label,out_per_record in [('empirical_mean',float(outputs.mean())),('empirical_p90',float(np.quantile(outputs,.9))),('max_output_8192_scenario',8192.)]:
        for period,price in rates.items():
            input_total=sum(token_inputs);output_total=len(ids)*out_per_record
            cost.append(dict(scenario=label,period=period,records=len(ids),
                estimated_input_tokens=input_total,estimated_output_tokens=round(output_total),
                assumed_cache='全部按缓存未命中计价',
                estimated_cny=round((input_total*price['input_miss']+output_total*price['output'])/1e6,2),
                note='单次成功请求情景；输入token为经验估计，重试与超时后已计费请求另计，不是硬上限'))
    with (DEST/'budget_scenarios.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cost[0]));w.writeheader();w.writerows(cost)
    paths={str(p):file_digest(p) for p in [INVENTORY,run_path(cfg)/'prepared.jsonl',run_path(cfg)/'extractions.jsonl',run_path(cfg)/'partials.jsonl']}
    summary=dict(records=len(ids),document_groups=len({r['document_group'] for r in selected}),
        common_candidate_pool=len(common),strata=dict(Counter(s for r in selected for s in r['strata'])),
        potential_company_years={str(ey):len({(r['source']['Symbol'],y) for r in selected
            for y in json.loads(common[r['record_id']]['candidate_protocols'])[f'{ey}:retrospective_snapshot2023']}) for ey in (2020,2021)},
        empirical_response_count=len(empirical),prompt_token_per_serialized_char_median=ratio,
        costs=cost,prices_cny_per_million=rates,price_checked_on='2026-10-15',
        price_source='https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
        price_evidence='官方文档搜索结果正文；直接打开该页超时。执行前复核账户实时价格。',
        price_time_rule='北京工作日09—12及14—18高峰；其余空闲，具体以官方规则为准',
        source_files_sha256=paths,api_calls=0,training_runs=0,status='待确认费用后调用；本轮只准备')
    write_json(DEST/'batch1_manifest.json',summary)
    content=(P/'configs/pilot.toml').read_text()
    content=content.replace('run_id = "pilot-20261015-snapshot2023-v1"','run_id = "auxtrain-b1-20261015-snapshot2023-v1"')
    content=content.replace('sample_size = 100',f'sample_size = {len(ids)}').replace('holdout_fraction = 0.2','holdout_fraction = 0.0')
    content+='\n# 只使用已保存的训练期候选ID；prepare不会重抽样。\n[selection]\nrecord_ids_file = "artifacts/followup/supervision_setup/batch1_record_ids.json"\n'
    (P/'configs/supervision_batch1.toml').write_text(content)
    assert paths=={s:file_digest(Path(s)) for s in paths}
    print(json.dumps({k:summary[k] for k in ['records','document_groups','common_candidate_pool','potential_company_years','costs','api_calls']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
