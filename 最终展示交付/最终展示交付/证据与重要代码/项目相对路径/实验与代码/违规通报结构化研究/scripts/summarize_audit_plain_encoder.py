"""Independent descriptive statistics for the predeclared ordinary X3 reference."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/experiments/20261016-evidence-v1/audit_information/plain_encoder'
BRANCH=ROOT.parent/'distributed'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def main():
 assert (ROOT/'verification.json').exists() and (BRANCH/'verification.json').exists()
 runs=json.loads((ROOT/'runs.json').read_text());gate=json.loads((ROOT/'pilot_gate.json').read_text())
 budget=json.loads((BRANCH/'capacity_selection.json').read_text())['selected_budget']
 years=sorted({r['year'] for r in runs});seeds=sorted({r['seed'] for r in runs})
 paths=[ROOT/n for n in ['runs.json','pilot_gate.json','verification.json','test_metrics.csv','predeclared_protocol.json']]
 paths+=[ROOT/'predeclared_protocol_v2.json',ROOT/'protocol.json']
 paths+=[BRANCH/n for n in ['capacity_selection.json','pilot_gate.json','test_metrics.csv']]
 before={str(p):sha(p) for p in paths}
 data=pd.read_csv(ROOT/'test_metrics.csv');branch=pd.read_csv(BRANCH/'test_metrics.csv')
 fields=['recall','precision','overall_ap','overall_logloss','overall_brier','conditional_logloss','conditional_brier','type_hit2','type_macro_coverage']
 yearly=data.groupby(['year','arm'])[fields].mean().reset_index();macro=data.groupby('arm')[fields].mean().reset_index()
 yearly.to_csv(ROOT/'year_arm_means.csv',index=False);macro.to_csv(ROOT/'macro_arm_means.csv',index=False)
 ref=branch[(branch.budget==budget)&branch.seed.isin(seeds)&branch.year.isin(years)&(branch.arm=='UA')]
 assert len(ref)==len(years)*len(seeds)
 candidate=data[data.arm=='UA'];assert len(candidate)==len(ref)
 cells=[]
 for year in years:
  for seed in seeds:
   a=candidate[(candidate.year==year)&(candidate.seed==seed)].iloc[0];b=ref[(ref.year==year)&(ref.seed==seed)].iloc[0]
   cells.append(dict(year=year,seed=seed,plain_UA_recall=float(a.recall),branch_UA_recall=float(b.recall),difference_pp=100*float(a.recall-b.recall)))
 summary=dict(runs=len(runs),budget=budget,seeds=seeds,stage='six_seed_reference' if len(seeds)==6 else 'two_seed_pilot_only',plain_hidden=int(runs[0]['hidden']),branch_X3_parameters=int(runs[0]['branch_X3_parameters']),plain_X3_parameters=int(runs[0]['plain_X3_parameters']),X3_parameter_difference=int(runs[0]['node_parameter_difference']),year_arm_means=yearly.to_dict('records'),macro_arm_means=macro.to_dict('records'),matched_seed_branch_UA_means=ref[fields].mean().to_dict(),plain_UA_minus_matched_branch_UA=dict(cells=cells,year_difference_pp={str(y):float(np.mean([r['difference_pp'] for r in cells if r['year']==y])) for y in years},macro_difference_pp=float(np.mean([r['difference_pp'] for r in cells]))),gate=gate,original_capacity_selection_changed=False,width_search_performed=False,bootstrap_computed=False,interpretation='Descriptive same-seed ordinary-encoder reference; two-seed pilot is not evidence of a robust architecture advantage. Does not alter the three prespecified primary branch bootstrap comparisons.',training_seconds_sum=sum(r['train_seconds'] for r in runs),source_sha256=before,predeclared_scientific_plan=str(ROOT/'predeclared_protocol.json'),pretraining_implementation_correction=str(ROOT/'predeclared_protocol_v2.json'))
 assert before=={p:sha(p) for p in before}
 save(ROOT/'summary.json',summary)
 save(ROOT/'statistics_verification.json',dict(source_metrics_gates_unchanged=True,new_training_runs=0,comparison_uses_same_branch_seeds=True,primary_capacity_selection_unchanged=True,source_sha256=before))
 print(json.dumps({k:summary[k] for k in ['runs','stage','budget','plain_hidden','X3_parameter_difference','plain_UA_minus_matched_branch_UA']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
