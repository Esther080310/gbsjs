"""Post-gate Global audit-category statistics; no fitting or capacity selection."""
from pathlib import Path
import hashlib,json,math,sys
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
import numpy as np
import pandas as pd
import summarize_evidence_20261016 as shared

def save_json(path,value):
 Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

# Process-local serialization adapter: the first passing comparison may return np.bool_.
# The shared statistics source remains frozen.
shared.write_json=save_json

ROOT=P/'artifacts/experiments/20261016-evidence-v1/audit_information/global'
FEATURES=ROOT.parent/'features'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 shared.check_weighted_topk()
 source_paths=[ROOT/'runs.json',ROOT/'pilot_gate.json',ROOT/'test_metrics.csv']
 source_paths+=list(ROOT.glob('*/*/predictions.npz'))
 source_paths+=[FEATURES/f'audit_features_{y}.npz' for y in [2020,2021]]
 source_paths+=[P.parent/f'财务年度配对修正/cache/{y}.npz' for y in [2020,2021]]
 before={str(p):sha(p) for p in source_paths}
 result=shared.summarize(ROOT,[('UA','UM'),('UA','U0')],1000)
 for comparison in result['comparisons']:
  comparison['stable_development_gain']=comparison.pop('continue_rule')
 result['interpretation']='Current-snapshot retrospective Global source-information comparison; stable_development_gain is not a prerequisite or hard gate for distributed capacity experiments.'
 result['distributed_experiments_constrained_by_global_gate']=False
 result['statistical_source_sha256']=before
 shared.write_json(ROOT/'summary.json',result)
 rows=[];coverage=[]
 for year in result['years']:
  with np.load(P.parent/f'财务年度配对修正/cache/{year}.npz',allow_pickle=False) as z:y=z['3_y'].astype(np.int64)
  with np.load(FEATURES/f'audit_features_{year}.npz',allow_pickle=False) as z:
   masks={'matched_IAR':z['3_iar_cohort'].copy(),'matched_IAR_and_audit_date':z['3_iar_and_auditdate_cohort'].copy()}
  assert np.all(~masks['matched_IAR_and_audit_date'] | masks['matched_IAR'])
  for cohort,mask in masks.items():
   assert mask.dtype==bool and len(mask)==len(y)
   coverage.append(dict(year=year,cohort=cohort,n=int(mask.sum()),positives=int(y[mask].sum()),within_cohort_budget=math.ceil(.1*mask.sum()),full_sample_n=len(y),full_sample_budget=math.ceil(.1*len(y))))
  for seed in result['seeds']:
   for arm in ['U0','UM','UA']:
    with np.load(ROOT/str(year)/f'{arm}_s{seed}/predictions.npz',allow_pickle=False) as z:score=z['a_logit'].copy()
    full_order=np.argsort(-score,kind='stable');full_selected=np.zeros(len(y),dtype=bool);full_selected[full_order[:math.ceil(.1*len(y))]]=True
    for cohort,mask in masks.items():
     ids=np.flatnonzero(mask);n=len(ids);positives=int(y[ids].sum());k=math.ceil(.1*n)
     suborder=np.argsort(-score[ids],kind='stable');hits=int(y[ids[suborder[:k]]].sum())
     fixed_ids=full_selected & mask;fixed_hits=int(y[fixed_ids].sum())
     rows.append(dict(year=year,seed=seed,arm=arm,cohort=cohort,n=n,positives=positives,within_cohort_budget=k,within_cohort_hits=hits,within_cohort_recall=hits/positives,within_cohort_precision=hits/k,fixed_full_list_budget=int(full_selected.sum()),fixed_full_list_cohort_selected=int(fixed_ids.sum()),fixed_full_list_cohort_hits=fixed_hits,fixed_full_list_cohort_positive_recall=fixed_hits/positives))
 frame=pd.DataFrame(rows);frame.to_csv(ROOT/'cohort_metrics.csv',index=False)
 numeric=[c for c in frame if c not in ['year','seed','arm','cohort']]
 yearly=frame.groupby(['year','cohort','arm'])[numeric].mean().reset_index();macro=frame.groupby(['cohort','arm'])[numeric].mean().reset_index()
 yearly.to_csv(ROOT/'cohort_year_arm_means.csv',index=False);macro.to_csv(ROOT/'cohort_macro_arm_means.csv',index=False)
 diffs=[]
 for cohort in masks:
  for candidate,reference in [('UA','UM'),('UA','U0')]:
   cells=[]
   for year in result['years']:
    for seed in result['seeds']:
     a=frame[(frame.year==year)&(frame.seed==seed)&(frame.arm==candidate)&(frame.cohort==cohort)].iloc[0]
     b=frame[(frame.year==year)&(frame.seed==seed)&(frame.arm==reference)&(frame.cohort==cohort)].iloc[0]
     assert a.n==b.n and a.positives==b.positives
     cells.append(dict(year=year,seed=seed,within_cohort_recall_difference_pp=100*(a.within_cohort_recall-b.within_cohort_recall),fixed_full_list_cohort_positive_recall_difference_pp=100*(a.fixed_full_list_cohort_positive_recall-b.fixed_full_list_cohort_positive_recall)))
   diffs.append(dict(cohort=cohort,candidate=candidate,reference=reference,cells=cells,macro_within_cohort_recall_difference_pp=float(np.mean([r['within_cohort_recall_difference_pp'] for r in cells])),macro_fixed_full_list_cohort_positive_recall_difference_pp=float(np.mean([r['fixed_full_list_cohort_positive_recall_difference_pp'] for r in cells]))))
 shared.write_json(ROOT/'cohort_summary.json',dict(coverage=coverage,comparisons=diffs,cohort_masks_shared_across_all_arms=True,ranking='Stable rank in original cache order; fresh 10% list inside each common cohort; separately retained original full-sample 10% list',confidence_intervals='Main full-sample comparisons use 1000 company-cluster draws; cohorts here report prespecified point sensitivity only',interpretation='Date cohorts are common posthoc source-metadata sensitivity sets, never feature filters or proof of historical first-release versions. Neither cohort selects distributed capacity.'))
 assert before=={p:sha(p) for p in before}
 shared.write_json(ROOT/'statistics_verification.json',dict(training_or_prediction_inputs_unchanged=True,gate_unchanged=True,new_training_runs=0,bootstrap_repetitions=1000,source_sha256=before,stable_gain_is_not_distributed_hard_gate=True))
 print(json.dumps({'comparisons':[dict(pair=x['candidate']+'-'+x['reference'],difference_pp=x['macro_mean_difference_pp'],CI_pp=[x['company_cluster_bootstrap']['percentile_95_lower_pp'],x['company_cluster_bootstrap']['percentile_95_upper_pp']],stable_development_gain=x['stable_development_gain']['passed']) for x in result['comparisons']],'cohorts':coverage,'cohort_differences':[{k:v for k,v in r.items() if k!='cells'} for r in diffs]},ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),flush=True)

if __name__=='__main__':main()
