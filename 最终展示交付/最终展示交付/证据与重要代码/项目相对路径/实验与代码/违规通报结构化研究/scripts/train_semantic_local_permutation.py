"""Prespecified local C_E/D_E diagnostic on the fixed changed-text support only."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time,tomllib
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('local_semantic_shared_driver',P/'scripts/train_auxiliary_semantic.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
import numpy as np
import pandas as pd
import torch
from notice_pipeline.permutation import prepare_case_permutation
from notice_pipeline.storage import file_digest

PROTOCOL='retrospective_snapshot2023'
ARMS=['C_E','D_E']
ROOT=P/'artifacts/experiments/20261016-evidence-v1/semantic_local_permutation'
DIRECTORY=P/'artifacts/followup/supervision_targets'


def read_csv(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def audit_and_aux(packs):
 targets=[json.loads(x) for x in (DIRECTORY/'targets.jsonl').read_text().splitlines()]
 by_index={t['target_index']:t for t in targets};by_id={t['target_id']:t for t in targets}
 provenance=read_csv(DIRECTORY/'provenance.csv')
 saved=read_csv(DIRECTORY/'permutation_audit/mapping.csv')
 # Independently reproduce the existing seed20261015 mapping solely as an integrity check.
 # Training donor indices always come from the saved mapping, not a newly selected permutation.
 check=prepare_case_permutation(targets,provenance,seed=20261015)
 assert len(saved)==len(check['mappings'])
 for a,b in zip(saved,check['mappings'],strict=True):
  assert all(a[k]==str(v) for k,v in b.items()),'Stored permutation mapping changed'
 blocks={b['block_id']:b for b in check['blocks']}
 facts=read_csv(P/'artifacts/followup/fact_roles.csv')
 fact_by_id={(x['record_id'],x['fact_id']):x for x in facts}
 holdout_docs={x['document_group'] for x in facts if x['pilot_split']=='holdout' or x.get('pilot_holdout_group')=='True'}
 docs={t['target_id']:set() for t in targets}
 for pr in provenance:
  docs[pr['target_id']].add(pr['document_group'])
  fact=fact_by_id[pr['record_id'],pr['fact_id']]
  assert fact['pilot_split']!='holdout' and fact.get('pilot_holdout_group')!='True'
  assert pr['document_group'] not in holdout_docs
 support=[];auxes={};used_mapping=[]
 with np.load(DIRECTORY/'permutation_audit/masks.npz',allow_pickle=False) as maskfile:
  np.testing.assert_array_equal(maskfile['target_index'],np.arange(len(targets)))
  np.testing.assert_array_equal(maskfile['target_id'],np.array([t['target_id'] for t in targets]))
  for key in ['D_source_target_index','can_exchange_mask','changed_text_mask']:
   dtype=bool if key.endswith('_mask') else int
   vals=np.array([r[key]=='True' if dtype is bool else int(r[key]) for r in saved],dtype=dtype)
   np.testing.assert_array_equal(vals,maskfile[key])
 for year,pack in packs.items():
  train=pack['splits'][0];aux=shared.load_auxiliary_vectors(DIRECTORY,year,PROTOCOL,0,train['keys'].Symbol.to_numpy(),train['keys'].economic_year.to_numpy())
  rows=[r for r in saved if int(r['evaluation_year'])==year and r['protocol']==PROTOCOL]
  assert sorted(int(r['D_source_target_index']) for r in rows)==sorted(int(r['target_index']) for r in rows),'Partition not bijective'
  eligible=[r for r in rows if r['changed_text_mask']=='True'];assert len(eligible)==21
  assert sorted(int(r['D_source_target_index']) for r in eligible)==sorted(int(r['target_index']) for r in eligible),'E mapping not bijective'
  mask=np.zeros(len(train['y']),dtype=bool);ci=np.full(len(mask),-1,dtype=np.int64);di=ci.copy();companies=set();documents=set();case_ids=set();signature_ids=set()
  for r in eligible:
   ti=int(r['target_index']);donor=int(r['D_source_target_index']);t=by_index[ti];d=by_index[donor];i=t['row_index']
   assert t['fold']==d['fold']==0 and t['protocol']==d['protocol']==PROTOCOL and t['evaluation_year']==d['evaluation_year']==year
   assert t['economic_year']==d['economic_year'] and t['B_coarse_text']==d['B_coarse_text']
   assert r['can_exchange_mask']=='True' and r['block_id']!=r['D_source_block_id']
   block=blocks[r['block_id']];db=blocks[r['D_source_block_id']]
   assert block['signature_id']==db['signature_id']==r['signature_id']
   assert set(block['companies']).isdisjoint(db['companies']) and set(block['document_groups']).isdisjoint(db['document_groups'])
   assert t['symbol']!=d['symbol'] and docs[t['target_id']].isdisjoint(docs[d['target_id']])
   assert docs[t['target_id']].isdisjoint(holdout_docs) and docs[d['target_id']].isdisjoint(holdout_docs)
   assert aux['aux_mask'][i] and aux['target_index'][i]==ti
   assert train['keys'].Symbol.iloc[i]==t['symbol'] and int(train['keys'].economic_year.iloc[i])==t['economic_year'] and train['y'][i]==1
   assert aux['aux_mask'][d['row_index']] and aux['target_index'][d['row_index']]==donor
   assert t['C_behavior_text']!=d['C_behavior_text']
   assert not np.array_equal(aux['C_vectors'][ti],aux['C_vectors'][donor]),'Distinct texts have identical actual target vectors'
   mask[i]=True;ci[i]=ti;di[i]=donor;companies.add(t['symbol']);documents.update(docs[t['target_id']]);case_ids.add(r['block_id']);signature_ids.add(r['signature_id'])
   used_mapping.append(dict(evaluation_year=year,row_index=i,symbol=t['symbol'],economic_year=t['economic_year'],target_index=ti,D_source_target_index=donor,D_source_symbol=d['symbol'],block_id=r['block_id'],D_source_block_id=r['D_source_block_id'],signature_id=r['signature_id']))
  assert mask.sum()==21 and not np.any(mask & ~aux['aux_mask'])
  # Independently inspect only held-out auxiliary masks, not their labels, inputs or metrics.
  with np.load(DIRECTORY/f'{year}_{PROTOCOL}.npz',allow_pickle=False) as af:
   for fold in [1,2,3]:assert not af[f'{fold}_aux_mask'].any()
  ce=dict(aux,aux_mask=mask.copy(),target_index=ci);de=dict(aux,aux_mask=mask.copy(),target_index=di)
  np.testing.assert_array_equal(ce['aux_mask'],de['aux_mask'])
  auxes[year]={'C_E':ce,'D_E':de}
  support.append(dict(evaluation_year=year,protocol=PROTOCOL,E_rows=int(mask.sum()),all_eligible_rows=int(aux['aux_mask'].sum()),independent_companies=len(companies),independent_case_blocks=len(case_ids),candidate_document_groups=len(documents),signature_count=len(signature_ids),row_bijection=True,whole_block_mapping_same_signature=True,cross_company_and_document_blocks=True,all_actual_vectors_changed=True,holdout_isolated=True,non_E_eligible_aux_disabled=True))
 global_targets=[by_index[r['target_index']] for r in used_mapping]
 return auxes,dict(unique_companies_across_evaluation_years=len({t['symbol'] for t in global_targets}),unique_company_years_across_evaluation_years=len({(t['symbol'],t['economic_year']) for t in global_targets}),unique_documents_across_evaluation_years=len(set().union(*(docs[t['target_id']] for t in global_targets))),partition_support=support,fixed_mapping_seed=20261015,existing_mapping_reproduced_exactly=True,holdout_document_groups=len(holdout_docs),mapping_not_a_randomization_distribution=True,interpretation='Correspondence diagnostic only on fixed small exchangeable support; never compare C_full against D_E'),used_mapping


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
 ROOT.mkdir(parents=True,exist_ok=True)
 gate_path=ROOT/'pilot_gate.json'
 assert not gate_path.exists(),'Existing gate; do not rerun or replace this fixed experiment'
 admission=P/'artifacts/experiments/20261016-evidence-v1/semantic/pilot_gate.json'
 admitted=json.loads(admission.read_text());assert admitted['C_vs_B_local_diagnostic_gate']['passed'] and admitted['test_metrics_opened'] is False
 cfg=tomllib.loads((P/'configs/semantic_20261016.toml').read_text())
 assert cfg['model']['head_seed_offset']==1000 and cfg['model']['auxiliary_weight']==.1
 assert cfg['training']['clip_backbone_norm']==cfg['training']['clip_head_norm']==1.
 assert cfg['study']['evaluation_years']==[2020,2021] and cfg['study']['pilot_seeds']==[42,100]
 assert cfg['study']['time_protocol']==PROTOCOL
 cfg['study'].update(arms=ARMS,output=str(ROOT.relative_to(P)))
 paths=[Path(__file__),P/'scripts/train_auxiliary_semantic.py',P/'configs/semantic_20261016.toml',P/'docs/20261016_数据后评估与实验协议.md',P/'notice_pipeline/alignment.py',P/'notice_pipeline/supervision.py',P/'notice_pipeline/permutation.py',P/'artifacts/followup/fact_roles.csv',DIRECTORY/'targets.jsonl',DIRECTORY/'provenance.csv',DIRECTORY/'embeddings/embeddings.npz',DIRECTORY/'embeddings/encoder_manifest.json',admission]
 paths+=list((DIRECTORY/'permutation_audit').glob('*'))
 paths+=[DIRECTORY/f'{year}_{PROTOCOL}.npz' for year in [2020,2021]]
 paths+=[shared.E/f'财务年度配对修正/cache/{year}{ext}' for year in [2020,2021] for ext in ['.npz','.json']]
 before={str(p):file_digest(p) for p in paths if p.is_file()}
 shared.write_json(ROOT/'source_hashes.json',before)
 shared.write_json(ROOT/'protocol.json',dict(shared_configuration=cfg,arms=ARMS,auxiliary_support='Exactly original changed_text_mask E, 21 per year',gate_comparison='C_E minus D_E',admission_file=str(admission),admission_sha256=file_digest(admission),no_new_permutation=True))
 packs={year:shared.load_pack(year,range(3)) for year in [2020,2021]}
 auxes,audit,mapping=audit_and_aux(packs)
 shared.write_json(ROOT/'support_audit.json',audit);pd.DataFrame(mapping).to_csv(ROOT/'used_mapping.csv',index=False)
 print('SUPPORT_AUDIT '+json.dumps(audit,ensure_ascii=False),flush=True)
 if args.audit_only:return
 loader=shared.load_loader();formal=loader.formal;torch.set_num_threads(cfg['training']['threads']);records=[];started=time.perf_counter()
 def run_seeds(seeds):
  for year,pack in packs.items():
   hashes={'cache':before[str(shared.E/f'财务年度配对修正/cache/{year}.npz')],'cache_metadata':before[str(shared.E/f'财务年度配对修正/cache/{year}.json')],'targets':before[str(DIRECTORY/'targets.jsonl')],'embeddings':before[str(DIRECTORY/'embeddings/embeddings.npz')],'existing_D_mapping':before[str(DIRECTORY/'permutation_audit/mapping.csv')]}
   for seed in seeds:
    for arm in ARMS:
     rec=shared.train_one(formal,pack,auxes[year][arm],year,seed,arm,ROOT/str(year)/f'{arm}_s{seed}',cfg,hashes)
     assert rec['auxiliary_mask_rows']==21
     records.append(rec);shared.write_json(ROOT/'runs.json',records)
     print(json.dumps({'year':year,'seed':seed,'arm':arm,'selection_loss':rec['selection_loss'],'epochs':rec['epochs']},ensure_ascii=False),flush=True)
    shared.verify_arm_randomness(ROOT,year,seed,ARMS)
 run_seeds(cfg['study']['pilot_seeds'])
 decision=shared.gate(records,'C_E','D_E',cfg['study']['evaluation_years'],cfg['study']['pilot_seeds'])
 result=dict(stage='initial_8_runs',decision=decision,expand_both_local_arms=decision['passed'],expansion_seeds=cfg['study']['expansion_seeds'] if decision['passed'] else [],test_metrics_opened=False,lambda_search_performed=False,small_support_warning='21 E rows per year; fixed single mapping, no randomization distribution')
 shared.write_json(gate_path,result);print('PILOT_GATE '+json.dumps(result),flush=True)
 if decision['passed']:run_seeds(cfg['study']['expansion_seeds'])
 shared.write_json(ROOT/'training_complete.json',dict(runs=len(records),pilot_gate_sha256=file_digest(gate_path),test_opened=False,elapsed_seconds_before_test=time.perf_counter()-started))
 shared.evaluate_after_gate(formal,ROOT,records,gate_path)
 assert before=={p:file_digest(Path(p)) for p in before},'Frozen source changed'
 shared.write_json(ROOT/'verification.json',dict(source_inputs_and_frozen_caches_unchanged=True,shared_E_masks_verified=True,same_backbone_head_rng_and_batch_order_verified=True,test_only_after_gate=True,runs=len(records),expanded=decision['passed'],elapsed_seconds=time.perf_counter()-started))
 print('COMPLETE_LOCAL',len(records),flush=True)

if __name__=='__main__':main()
